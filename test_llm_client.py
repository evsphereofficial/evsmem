"""Tests for llm_client: DeepSeek remote client + local GGUF fallback.

Run from the evsmem directory:
    .venv\\Scripts\\python.exe -m unittest test_llm_client -v

Covers: retry/backoff, rate-limit handling, timeout, JSON errors, fallback to
the local LLMClient, and OpenAI-style function-calling round trips.
"""

import io
import json
import socket
import sys
import unittest
import urllib.error
from unittest import mock

import llm_client


class _FakeResponse:
    """Minimal stand-in for urllib's HTTPResponse."""

    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHTTPError(urllib.error.HTTPError):
    """HTTPError that behaves like urllib's but is easy to construct."""

    def __init__(self, code, reason="error"):
        super().__init__("https://opencode.ai/zen/v1/chat/completions", code, reason, None, None)


def _json_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _chat_payload(content=None, tool_calls=None) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []}}]}


class _FakeLocalLLM:
    """LLMClient-shaped local fallback returning a canned string."""

    def __init__(self, text="LOCAL_REPLY"):
        self._text = text
        self.calls = 0

    def generate(self, messages, max_tokens=512, temperature=0.1):
        self.calls += 1
        return self._text


class RetryAndFallbackTest(unittest.TestCase):
    """Retry logic: transient failures retried with backoff, then local fallback."""

    def setUp(self):
        self.client = llm_client.DeepSeekClient(
            api_key="test-key",
            base_url="https://opencode.ai/zen/v1",
            fallback=_FakeLocalLLM(),
        )
        self.sleep_patch = mock.patch.object(llm_client.time, "sleep", return_value=None)
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_generate_success(self):
        with mock.patch.object(llm_client, "_urlopen", return_value=_FakeResponse(
            200, _json_body(_chat_payload(content="REMOTE_REPLY")),
        )) as urlopen:
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "REMOTE_REPLY")
        urlopen.assert_called_once()
        # Request carries Authorization header derived from the env key.
        req = urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Authorization"), "Bearer test-key")

    def test_retry_on_429_then_success(self):
        responses = [_FakeHTTPError(429), _FakeResponse(200, _json_body(_chat_payload(content="OK")))]
        with mock.patch.object(llm_client, "_urlopen", side_effect=responses) as urlopen:
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "OK")
        self.assertEqual(urlopen.call_count, 2)

    def test_retry_on_500_then_local_fallback(self):
        responses = [_FakeHTTPError(503), _FakeHTTPError(503), _FakeHTTPError(503), _FakeHTTPError(503)]
        with mock.patch.object(llm_client, "_urlopen", side_effect=responses) as urlopen:
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "LOCAL_REPLY")
        self.assertEqual(self.client._local.calls, 1)

    def test_timeout_then_local_fallback(self):
        with mock.patch.object(llm_client, "_urlopen", side_effect=socket.timeout("timed out")) as urlopen:
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "LOCAL_REPLY")

    def test_network_error_then_local_fallback(self):
        with mock.patch.object(llm_client, "_urlopen", side_effect=urllib.error.URLError("no route")):
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "LOCAL_REPLY")

    def test_auth_error_immediate_local_fallback(self):
        with mock.patch.object(llm_client, "_urlopen", side_effect=_FakeHTTPError(401, "Unauthorized")) as urlopen:
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "LOCAL_REPLY")
        self.assertEqual(urlopen.call_count, 1)  # 401 is not retried

    def test_json_error_then_local_fallback(self):
        with mock.patch.object(llm_client, "_urlopen", return_value=_FakeResponse(200, b"not json{{{")):
            out = self.client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "LOCAL_REPLY")

    def test_missing_api_key_uses_local_directly(self):
        client = llm_client.DeepSeekClient(api_key="", fallback=_FakeLocalLLM("LOCAL"))
        self.assertFalse(client.is_available())
        with mock.patch.object(llm_client, "_urlopen", side_effect=AssertionError("must not call HTTP")) as urlopen:
            out = client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "LOCAL")
        urlopen.assert_not_called()


class FunctionCallingTest(unittest.TestCase):
    """OpenAI-style function-calling round trips."""

    def setUp(self):
        self.client = llm_client.DeepSeekClient(
            api_key="test-key",
            base_url="https://opencode.ai/zen/v1",
            fallback=_FakeLocalLLM(),
        )
        self.sleep_patch = mock.patch.object(llm_client.time, "sleep", return_value=None)
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_generate_with_tools_returns_openai_shape(self):
        tool_call = {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        payload = _chat_payload(content=None, tool_calls=[tool_call])
        with mock.patch.object(llm_client, "_urlopen", return_value=_FakeResponse(200, _json_body(payload))):
            resp = self.client.generate_with_tools([{"role": "user", "content": "weather?"}],
                                                   tools=[{"type": "function", "function": {"name": "get_weather"}}])
        content, calls = llm_client._normalize_tool_response(resp)
        self.assertEqual(content, "")  # empty content normalized like deriver does
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertIn("Paris", calls[0]["function"]["arguments"])

    def test_full_round_trip_executes_tools_and_returns_final_answer(self):
        # First round: model asks for a tool. Second round: final answer.
        tool_call = {"id": "call_1", "type": "function",
                     "function": {"name": "add", "arguments": '{"a":2,"b":3}'}}
        payloads = [
            _json_body(_chat_payload(content=None, tool_calls=[tool_call])),
            _json_body(_chat_payload(content="the sum is 5")),
        ]
        executed = []

        def tool_executor(name, arguments):
            executed.append((name, arguments))
            return arguments["a"] + arguments["b"]

        with mock.patch.object(llm_client, "_urlopen", side_effect=[
            _FakeResponse(200, payloads[0]), _FakeResponse(200, payloads[1]),
        ]) as urlopen:
            out = self.client.chat_with_tools(
                [{"role": "user", "content": "2+3?"}],
                tools=[{"type": "function", "function": {"name": "add"}}],
                tool_executor=tool_executor,
            )
        self.assertEqual(out, "the sum is 5")
        self.assertEqual(executed, [("add", {"a": 2, "b": 3})])
        self.assertEqual(urlopen.call_count, 2)

    def test_generate_with_tools_falls_back_to_local_on_failure(self):
        with mock.patch.object(llm_client, "_urlopen", side_effect=_FakeHTTPError(500)):
            resp = self.client.generate_with_tools([{"role": "user", "content": "hi"}],
                                                   tools=[{"type": "function", "function": {"name": "x"}}])
        content, calls = llm_client._normalize_tool_response(resp)
        self.assertEqual(content, "LOCAL_REPLY")
        self.assertEqual(calls, [])
        self.assertEqual(self.client._local.calls, 1)


class FetchModelsTest(unittest.TestCase):
    """/zen/v1/models connectivity + model id resolution."""

    def setUp(self):
        self.sleep_patch = mock.patch.object(llm_client.time, "sleep", return_value=None)
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_fetch_available_models_parses_ids(self):
        body = {"data": [{"id": "zen/deepseek-v4-flash-free"}, {"id": "other-model"}]}
        with mock.patch.object(llm_client, "_urlopen", return_value=_FakeResponse(200, _json_body(body))):
            ids = llm_client.fetch_available_models()
        self.assertIn("zen/deepseek-v4-flash-free", ids)

    def test_fetch_available_models_inaccessible_returns_empty(self):
        with mock.patch.object(llm_client, "_urlopen", side_effect=_FakeHTTPError(403, "Forbidden")):
            ids = llm_client.fetch_available_models()
        self.assertEqual(ids, [])
        # Documented model id is still known → pipeline can proceed.
        self.assertTrue(llm_client.DEEPSEEK_MODEL)


if __name__ == "__main__":
    unittest.main()