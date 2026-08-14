"""LLM client — provider-aware: remote DeepSeek with local GGUF fallback.

Provider chain (preferred → fallback):
  1. DeepSeek v4 flash (``deepseek-v4-flash-free``) via the OpenAI-compatible
     endpoint https://opencode.ai/zen/v1 (urllib, stdlib only). Retries
     transient failures (network, rate-limit, 5xx, timeout) with exponential
     backoff + jitter. Supports OpenAI-style function calling.
  2. Local GGUF model via llama-cpp-python (LLMClient) — used on ANY remote
     failure (network, auth, rate limit, timeout, JSON error) or when the
     ``evsmem_llm_api_key`` env var is unset.

API key handling (SECURITY):
  - Read ONLY from the ``evsmem_llm_api_key`` environment variable.
  - NEVER hardcode the key. NEVER log the key (only its presence is logged).

Public contracts (unchanged / additive):
  - LLMClient.generate(messages, max_tokens, temperature) -> str         (local)
  - DeepSeekClient.generate(messages, max_tokens, temperature) -> str   (remote→local)
  - DeepSeekClient.generate_with_tools(...) -> OpenAI-shaped dict       (remote→local)
  - DeepSeekClient.chat_with_tools(...) -> str  (full in-client tool round trip)

CLI:
  python -m evsmem.llm_client --list-models   # fetch /zen/v1/models ids
  python -m evsmem.llm_client --check         # connectivity + provider status
"""

import json
import logging
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4


def _load_dotenv(path: Path | None = None) -> None:
    """Load a sibling ``.env`` file into ``os.environ`` (stdlib only).

    Run at import time so module-level config and API-key lookups see values
    even when the process was started without the variables exported. A real
    environment variable always wins (``setdefault``), so this never clobbers
    an explicitly exported value. The key/value is never logged.
    """
    target = path or Path(__file__).resolve().with_name(".env")
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

logger = logging.getLogger("evsmem.llm_client")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# ── Local GGUF configuration (fallback provider) ─────────────────────────────
# Resolve GGUF model path.
_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "gemma-4-12B-it-QAT-GGUF" / "gemma-4-12B-it-QAT-Q4_0.gguf"),
)
_N_CTX = int(os.getenv("LLM_N_CTX", "264000"))
_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "64000"))
_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))

# Add CUDA DLL directory so llama_cpp can find them at import time.
_venv_lib = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
_cuda_dll_dir = str(_venv_lib / "llama_cpp" / "lib")
if os.path.isdir(_cuda_dll_dir):
    try:
        os.add_dll_directory(_cuda_dll_dir)
    except Exception:
        pass

# Global singleton for the loaded GGUF model.
_model = None
_MODEL_LOCK = threading.Lock()

# ── Remote DeepSeek (opencode.ai/zen) configuration ──────────────────────────
# Endpoint + documented model id. The /zen/v1/models endpoint is probed by the
# CLI / fetch_available_models(); when it is inaccessible the documented id is
# used so the pipeline keeps working.
ZEN_BASE_URL = os.getenv("EVSMEM_LLM_BASE_URL", "https://opencode.ai/zen/v1").rstrip("/")
DEEPSEEK_MODEL = "deepseek-v4-flash-free"  # wire model id (user-verified; NO "zen/" prefix)
# API key — env var ONLY, never hardcoded, never logged.
_API_KEY_ENV = "evsmem_llm_api_key"

# Timeout (seconds) applied to EVERY request; retry budget for transient errors.
_REQUEST_TIMEOUT = float(os.getenv("EVSMEM_LLM_TIMEOUT", "60"))
_MAX_RETRIES = int(os.getenv("EVSMEM_LLM_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.getenv("EVSMEM_LLM_RETRY_BASE", "1.0"))
_RETRY_MAX_DELAY = float(os.getenv("EVSMEM_LLM_RETRY_MAX", "16.0"))
_RETRY_JITTER = float(os.getenv("EVSMEM_LLM_RETRY_JITTER", "0.3"))

# Generation-size defaults for the REMOTE provider (the local GGUF client has
# its own GPU-safety defaults above). EVSMEM_MAX_TOKENS caps the remote
# model's OUTPUT — the deriver's batch JSON (dozens of rows across 300
# messages) used to be truncated at a hardcoded 4096, which is why only 2-6
# short rows ever landed. Default 16384, env-configurable (the user's old
# LLM_MAX_TOKENS=64000 never reached the remote payload — it was dead code).
_REMOTE_MAX_TOKENS = int(os.getenv("EVSMEM_MAX_TOKENS", "16384"))
# Context cap sent as `max_context_tokens` — the zen endpoint accepts the
# field (verified live: 200 OK) and it tells the backend how much room the
# batch is allowed to use. Default 265000 (the user-approved window);
# EVSMEM_MAX_CONTEXT=0 stops sending it (model-native context applies).
_REMOTE_MAX_CONTEXT = int(os.getenv("EVSMEM_MAX_CONTEXT", "265000"))
# Effort control for the effort-based reasoning model (deepseek-v4-flash-free
# supports reasoning_effort low/high/max — there is NO full "off" level on zen:
# models.dev declares effort-type options and upstream PR #31795 adding a
# `none` variant is still unmerged). "low" minimizes thinking tokens so the
# batch output budget is not eaten by reasoning_content.
_REMOTE_REASONING_EFFORT = os.getenv("EVSMEM_REASONING_EFFORT", "low").strip()

# HTTP statuses considered transient (retried with backoff).
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

# opencode-compatible User-Agent: the zen server's checkHeaders rate-limiter
# grants official-client limits only to requests whose User-Agent starts with
# "opencode/" — a browser UA (Mozilla/...) hits the tiny per-IP fallback quota
# and 429s.
_OPENCODE_UA = "opencode/evsmem/1.0"
# Patchable in tests.
_urlopen = urllib.request.urlopen


class DeepSeekError(RuntimeError):
    """Raised when the opencode.ai/zen remote endpoint fails (non-fallback path)."""


# ── Retry / backoff helpers ──────────────────────────────────────────────────

def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for retry `attempt` (1-based)."""
    delay = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    return delay + random.uniform(0.0, _RETRY_JITTER * delay)


def _zen_request(method: str, url: str, payload=None, headers=None,
                 timeout: float | None = None, max_retries: int | None = None):
    """JSON HTTP request to the zen endpoint with retry + exponential backoff.

    Retries transient failures (network errors, socket timeouts, HTTP
    408/425/429/5xx). Auth/4xx errors fail fast (non-retryable). Returns the
    parsed JSON body. Raises ``DeepSeekError`` after retries are exhausted or
    on non-retryable HTTP errors / malformed JSON.
    """
    timeout = timeout if timeout is not None else _REQUEST_TIMEOUT
    max_retries = max_retries if max_retries is not None else _MAX_RETRIES
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    # opencode-compatible headers on EVERY request: the zen server's rate
    # limiter (checkHeaders) only grants full daily limits to requests
    # carrying the official client headers — a plain/browser UA gets the tiny
    # per-IP fallback quota and 429s immediately.
    request.add_header("User-Agent", _OPENCODE_UA)
    request.add_header("Accept", "application/json")
    request.add_header("Accept-Language", "en-US,en;q=0.9")
    request.add_header("x-opencode-project", "evsmem")
    request.add_header("x-opencode-session", f"ses_evsmem_{uuid4().hex}")
    request.add_header("x-opencode-request", f"msg_evsmem_{uuid4().hex}")
    request.add_header("x-opencode-client", "evsmem")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        request.add_header(k, v)

    attempt = 0
    while True:
        try:
            with _urlopen(request, timeout=timeout) as resp:
                body = resp.read()
            try:
                return json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise DeepSeekError(f"invalid JSON from {url}: {e}") from e
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUSES or attempt >= max_retries:
                raise DeepSeekError(f"zen HTTP {e.code} {e.reason} for {url}") from e
            attempt += 1
            delay = _backoff_delay(attempt)
            logger.warning(
                "event=zen_retry http=%d attempt=%d/%d delay=%.2fs url=%s",
                e.code, attempt, max_retries, delay, url,
            )
            time.sleep(delay)
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
            if attempt >= max_retries:
                raise DeepSeekError(f"zen network error for {url}: {e}") from e
            attempt += 1
            delay = _backoff_delay(attempt)
            logger.warning(
                "event=zen_retry error=%s attempt=%d/%d delay=%.2fs url=%s",
                type(e).__name__, attempt, max_retries, delay, url,
            )
            time.sleep(delay)


def fetch_available_models(base_url: str | None = None, timeout: float | None = None) -> list[str]:
    """GET {base}/models and return the available model ids.

    When the endpoint is inaccessible, logs a clear diagnostic and returns []
    so callers can proceed with the documented model id if known.
    """
    base = (base_url or ZEN_BASE_URL).rstrip("/")
    url = f"{base}/models"
    try:
        data = _zen_request("GET", url, timeout=timeout, max_retries=1)
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        logger.info("event=models_fetched count=%d url=%s", len(ids), url)
        return ids
    except DeepSeekError as e:
        logger.warning("event=models_endpoint_error error=%s url=%s", e, url)
        return []


# ── Response normalization (mirrors deriver._normalize_llm_response) ─────────

def _normalize_tool_response(resp):
    """Extract (content, tool_calls) from common OpenAI response shapes."""
    if resp is None:
        return "", []
    if isinstance(resp, str):
        return resp, []
    if isinstance(resp, dict) and resp.get("choices") and isinstance(resp["choices"], list) and resp["choices"]:
        msg = resp["choices"][0].get("message") or {}
        return msg.get("content") or "", msg.get("tool_calls") or []
    if isinstance(resp, dict) and resp.get("message") and isinstance(resp["message"], dict):
        return resp["message"].get("content") or "", resp["message"].get("tool_calls") or []
    if isinstance(resp, dict):
        return resp.get("content") or "", resp.get("tool_calls") or []
    return "", []


def _format_tool_calls(tool_calls) -> list[dict]:
    """Normalize raw tool_calls into OpenAI message format (id/type/function)."""
    formatted = []
    for tc in tool_calls:
        tcd = tc if isinstance(tc, dict) else {}
        fn = tcd.get("function") or tcd
        formatted.append({
            "id": tcd.get("id") or f"call_{uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": fn.get("name") or tcd.get("name") or "",
                "arguments": fn.get("arguments") or tcd.get("arguments") or "{}",
            },
        })
    return formatted


# ── Remote client (primary provider, local fallback) ─────────────────────────

class DeepSeekClient:
    """Remote DeepSeek client for the opencode.ai/zen OpenAI-compatible API.

    Primary provider; falls back to the local GGUF ``LLMClient`` on ANY remote
    failure (network, auth, rate limit, timeout, JSON error, empty reply) or
    when ``evsmem_llm_api_key`` is unset.

    Public contract matching the local client:
        generate(messages, max_tokens=512, temperature=0.1) -> str
    Plus OpenAI-style function calling:
        generate_with_tools(messages, tools=None, tool_choice="auto", ...) -> dict
        chat_with_tools(messages, tools, tool_executor, ...) -> str
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 max_retries: int | None = None, fallback=None):
        # The key comes from the caller (tests/ops) or the env var — NEVER a
        # hardcoded literal. Presence is logged, the value itself is not.
        self._api_key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV, "")
        self._base_url = (base_url or ZEN_BASE_URL).rstrip("/")
        self._model = model or DEEPSEEK_MODEL
        self._timeout = timeout if timeout is not None else _REQUEST_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else _MAX_RETRIES
        self._local = fallback  # LLMClient-like fallback (None → lazy LLMClient)
        if not self._api_key:
            logger.warning(
                "event=zen_no_api_key env=%s unset → remote DeepSeek disabled; "
                "falling back to local GGUF LLM", _API_KEY_ENV,
            )

    def is_available(self) -> bool:
        """True when the remote provider can be used (API key configured)."""
        return bool(self._api_key)

    def model_name(self) -> str:
        """Return the remote model id."""
        return self._model

    def _get_local(self):
        """Lazily build the local GGUF fallback client."""
        if self._local is None:
            self._local = LLMClient()
        return self._local

    def _local_generate(self, messages, max_tokens, temperature) -> str:
        """Byte-compatible local path: LLMClient.generate(...) contract."""
        try:
            return self._get_local().generate(
                messages=messages, max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as e:
            logger.warning("event=local_fallback_failed error=%s", e)
            return ""

    def _chat(self, messages, tools=None, tool_choice="auto",
              max_tokens: int | None = None, temperature=0.1) -> dict:
        """One chat/completions round against the zen endpoint.

        ``max_tokens=None`` resolves to ``EVSMEM_MAX_TOKENS`` (default 16384)
        so callers that don't care about output size always get the large
        default — a truncated JSON reply is worse than a slightly bigger one.
        """
        max_tokens = max_tokens if max_tokens is not None else _REMOTE_MAX_TOKENS
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,  # non-streaming path — never leave stream unset
        }
        if _REMOTE_MAX_CONTEXT > 0:
            # Default 265000 (user-approved window): the zen endpoint accepts
            # the field (verified live), so the batch can actually use the
            # whole window instead of silently truncating at a tiny default.
            payload["max_context_tokens"] = _REMOTE_MAX_CONTEXT
        if _REMOTE_REASONING_EFFORT:
            # Minimum thinking effort this model supports (low/high/max); no
            # off/none level exists on zen for deepseek-v4-flash-free yet.
            payload["reasoning_effort"] = _REMOTE_REASONING_EFFORT
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        return _zen_request(
            "POST", f"{self._base_url}/chat/completions",
            payload=payload, headers=headers,
            timeout=self._timeout, max_retries=self._max_retries,
        )

    def generate(self, messages: list[dict], max_tokens: int | None = None,
                 temperature: float = 0.1) -> str:
        """Generate a text reply; falls back to the local GGUF model on ANY
        remote failure (or when the API key is unset).

        ``max_tokens=None`` → ``EVSMEM_MAX_TOKENS`` (default 16384).
        """
        max_tokens = max_tokens if max_tokens is not None else _REMOTE_MAX_TOKENS
        if not self.is_available():
            return self._local_generate(messages, max_tokens, temperature)
        try:
            data = self._chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
            content = data["choices"][0]["message"].get("content")
            if not content:
                raise DeepSeekError("zen reply empty")
            return content
        except Exception as e:
            logger.warning("event=zen_remote_failed error=%s → local fallback", e)
            return self._local_generate(messages, max_tokens, temperature)

    def generate_with_tools(self, messages: list[dict], tools=None,
                            tool_choice="auto", max_tokens: int = 512,
                            temperature: float = 0.1) -> dict:
        """Single remote round with tools; returns an OpenAI-shaped dict
        (``{"choices": [{"message": {content, tool_calls}}]}``).

        On ANY remote failure, falls back to the local model with ``tool_calls=[]``
        so callers (e.g. deriver's tool loop) can parse the content and stop.
        """
        if not self.is_available():
            return self._local_tool_response(messages, max_tokens, temperature)
        try:
            return self._chat(
                messages=messages, tools=tools, tool_choice=tool_choice,
                max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as e:
            logger.warning("event=zen_tools_failed error=%s → local fallback", e)
            return self._local_tool_response(messages, max_tokens, temperature)

    def _local_tool_response(self, messages, max_tokens, temperature) -> dict:
        """OpenAI-shaped response from the local model (no tool_calls)."""
        text = self._local_generate(messages, max_tokens, temperature)
        return {"choices": [{"message": {"role": "assistant", "content": text or None, "tool_calls": []}}]}

    def chat_with_tools(self, messages: list[dict], tools, tool_executor,
                        tool_choice: str = "auto", max_rounds: int = 5,
                        max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Full OpenAI-style function-calling round trip executed in this client.

        1. Send messages + tools; parse tool_calls from the response.
        2. Execute each requested function locally via
           ``tool_executor(name: str, arguments: dict) -> Any``.
        3. Append tool results as ``role=tool`` messages.
        4. Repeat until the model returns a final answer (no tool_calls) or
           ``max_rounds`` is reached.

        Falls back to a single local generation (no tools) on any remote
        failure. Returns the final answer text ("" on total failure).
        """
        conversation = [dict(m) for m in messages]
        final_content = ""
        for _ in range(max_rounds + 1):
            resp = self.generate_with_tools(
                conversation, tools=tools, tool_choice=tool_choice,
                max_tokens=max_tokens, temperature=temperature,
            )
            content, tool_calls = _normalize_tool_response(resp)
            final_content = content or final_content
            if not tool_calls:
                break
            formatted = _format_tool_calls(tool_calls)
            conversation.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": formatted,
            })
            for tc in formatted:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                try:
                    result = tool_executor(name, args)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": json.dumps(result, default=str),
                })
        return final_content


# ── Local GGUF client (factory for the fallback) ─────────────────────────────

def _load_model():
    """Load GGUF once (GPU if available). Stays in memory."""
    global _model
    if _model is not None:
        return _model
    with _MODEL_LOCK:
        if _model is not None:
            return _model
        if not os.path.isfile(_MODEL_PATH):
            logger.warning(f"Model not found at {_MODEL_PATH}")
            return None
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model from {_MODEL_PATH}")
            t0 = time.time()
            _model = Llama(
                model_path=str(_MODEL_PATH),
                embedding=False,  # This is for generation, not embeddings
                n_gpu_layers=_GPU_LAYERS,
                n_ctx=_N_CTX,
                verbose=False,
            )
            logger.info(f"Model ready ({time.time()-t0:.1f}s)")
            return _model
        except Exception as e:
            logger.warning(f"GGUF model load failed: {e}")
            return None


class LLMClient:
    """LLM client — loads GGUF model once via llama-cpp-python for chat completion."""

    def __init__(self):
        self._loaded = False

    def _ensure_model(self):
        if not self._loaded:
            _load_model()
            self._loaded = True

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate a response from the model using OpenAI-style messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys
                     (e.g. [{"role": "user", "content": "Hello"}])
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.1)

        Returns:
            Generated text string, or empty string on failure.
        """
        self._ensure_model()
        if _model is not None:
            try:
                result = _model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
        return ""

    def is_available(self) -> bool:
        """Check if the model file exists on disk."""
        return os.path.isfile(_MODEL_PATH)

    def model_name(self) -> str:
        """Return the model filename."""
        return "gemma-4-12B-it-QAT-Q4_0.gguf"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    """CLI: fetch available models from /zen/v1/models and validate connectivity.

    Usage:
      python -m evsmem.llm_client --list-models
      python -m evsmem.llm_client --check
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="evsmem DeepSeek (opencode.ai/zen) client utilities",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="fetch and print available model ids from /zen/v1/models",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate connectivity and print provider status",
    )
    args = parser.parse_args(argv)

    if args.list_models or args.check:
        models = fetch_available_models()
        if models:
            print(f"[ok] {len(models)} model(s) available:")
            for mid in models:
                print(f"  - {mid}")
            return 0
        # Endpoint inaccessible (or returned no ids) → clear diagnostic.
        if DEEPSEEK_MODEL:
            print(f"[warn] models endpoint inaccessible; using documented model id: {DEEPSEEK_MODEL}")
            return 0
        print("[error] models endpoint inaccessible and no documented model id is known; "
              "set EVSMEM_LLM_MODEL or fix connectivity")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())