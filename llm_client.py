"""LLM client — LFM2.5-1.2B-Instruct GGUF via llama-cpp-python with CUDA."""

import logging, os, threading, time
from pathlib import Path

logger = logging.getLogger("evsmem.llm_client")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# Resolve GGUF model path
_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "gemma-4-12B-it-QAT-GGUF" / "gemma-4-12B-it-QAT-Q4_0.gguf"),
)
_N_CTX = int(os.getenv("LLM_N_CTX", "32768"))
_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16384"))
_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))

# Add CUDA DLL directory so llama_cpp can find them at import time
_venv_lib = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
_cuda_dll_dir = str(_venv_lib / "llama_cpp" / "lib")
if os.path.isdir(_cuda_dll_dir):
    try:
        os.add_dll_directory(_cuda_dll_dir)
    except Exception:
        pass

# Global singleton
_model = None
_MODEL_LOCK = threading.Lock()


def _load_model():
    """Load LFM2.5 GGUF once (GPU if available). Stays in memory."""
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
