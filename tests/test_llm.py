"""Устойчивость LLM-клиентов к сбоям провайдера. Без сети: SDK подменён заглушками."""

from __future__ import annotations

import httpx
import httpx2
import ollama
import pytest
from google.genai import errors, types
from pydantic import BaseModel

from ravel.llm import (
    LLMClient,
    LLMError,
    LLMUnavailable,
    QuotaExhausted,
    TransientError,
    TruncatedResponse,
    get_client,
)
from ravel.llm.base import provider_text
from ravel.llm.gemini import GeminiClient
from ravel.llm.ollama import OllamaClient

FAKE_KEY = "AIzaSyTEST_" + "k" * 28


class Verdict(BaseModel):
    ok: bool


class Findings(BaseModel):
    findings: list[str]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RAVEL_MAX_CALLS", "RAVEL_LLM_TIMEOUT", "RAVEL_MAX_OUTPUT_TOKENS", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)


# ---------------------------------------------------------------- helpers


def api_error(code: int, status: str, message: str, details=None, headers=None) -> errors.APIError:
    body = {"error": {"code": code, "message": message, "status": status}}
    if details is not None:
        body["error"]["details"] = details
    response = httpx2.Response(code, headers=headers) if headers else None
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, body, response)


def quota_429(quota_id: str, retry_delay: str = "7s", value: str = "20") -> errors.APIError:
    # форма ответа Gemini API при исчерпании квоты бесплатного тарифа
    return api_error(
        429,
        "RESOURCE_EXHAUSTED",
        f"You exceeded your current quota. Please retry in {retry_delay}.",
        details=[
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        "quotaId": quota_id,
                        "quotaValue": value,
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    )


def gemini_response(text: str | None = '{"ok": true}', finish: str | None = "STOP"):
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)]
    )


class Script:
    """Заглушка метода SDK: отдаёт ответы или бросает исключения по очереди, считает вызовы."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_gemini(*outcomes, **kw) -> tuple[GeminiClient, Script, list[float]]:
    client = GeminiClient("gemini-test", FAKE_KEY, None, **kw)
    script = Script(*outcomes)
    client._client.models.generate_content = script
    sleeps: list[float] = []
    client.retry_sleep = sleeps.append
    return client, script, sleeps


def ask(client: LLMClient, schema=Verdict, user: str = "u"):
    return client.complete("s", user, schema)


# ------------------------------------------------------------- happy path


def test_gemini_passes_limits_and_parses():
    client, script, _ = make_gemini(gemini_response(), max_output_tokens=2048, timeout=42)
    assert ask(client) == Verdict(ok=True)
    config = script.calls[0]["config"]
    assert config.max_output_tokens == 2048
    assert config.seed == 17 and config.temperature == 0
    assert config.response_schema is Verdict
    assert client._client._api_client._http_options.timeout == 42_000  # HttpOptions.timeout в мс
    assert client.calls == 1


def test_empty_findings_list_is_a_normal_answer():
    client, script, _ = make_gemini(gemini_response('{"findings": []}'))
    assert ask(client, Findings) == Findings(findings=[])
    assert len(script.calls) == 1


# ------------------------------------------------- permanent provider errors


@pytest.mark.parametrize(
    "error",
    [
        api_error(403, "PERMISSION_DENIED", "Your API key was reported as leaked."),
        api_error(401, "UNAUTHENTICATED", "Request had invalid authentication credentials."),
        api_error(
            400,
            "INVALID_ARGUMENT",
            "API key not valid. Please pass a valid API key.",
            details=[{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}],
        ),
        api_error(404, "NOT_FOUND", "models/gemini-x is not found for API version v1"),
    ],
    ids=["403", "401", "bad-key-400", "404"],
)
def test_auth_and_missing_model_are_unavailable_without_retries(error):
    client, script, sleeps = make_gemini(error)
    with pytest.raises(LLMUnavailable) as exc:
        ask(client)
    assert not isinstance(exc.value, QuotaExhausted)
    assert len(script.calls) == 1 and sleeps == []


def test_404_names_the_model_setting():
    client, _, _ = make_gemini(api_error(404, "NOT_FOUND", "not found"))
    with pytest.raises(LLMUnavailable, match="GEMINI_MODEL"):
        ask(client)


def test_plain_400_is_llm_error_not_unavailable():
    client, script, _ = make_gemini(
        api_error(400, "INVALID_ARGUMENT", "Request contains an invalid argument.")
    )
    with pytest.raises(LLMError) as exc:
        ask(client)
    assert not isinstance(exc.value, LLMUnavailable)
    assert len(script.calls) == 1


# ------------------------------------------------------------------ quota


def test_daily_quota_429_stops_at_once_and_blocks_next_calls():
    client, script, sleeps = make_gemini(quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    with pytest.raises(QuotaExhausted, match="PerDay"):
        ask(client)
    assert len(script.calls) == 1 and sleeps == []
    with pytest.raises(QuotaExhausted):
        ask(client, user="другой запрос")
    assert len(script.calls) == 1  # второй complete() не ходил к провайдеру
    assert client.calls == 1


def test_per_minute_429_retries_with_retry_delay_then_quota_exhausted():
    client, script, sleeps = make_gemini(quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))
    with pytest.raises(QuotaExhausted, match="попыток: 3"):
        ask(client)
    assert len(script.calls) == 3
    assert len(sleeps) == 2 and all(7 <= s <= 60 for s in sleeps)  # не меньше retryDelay
    with pytest.raises(QuotaExhausted):
        ask(client, user="ещё")
    assert len(script.calls) == 3


def test_429_then_success_recovers():
    client, script, sleeps = make_gemini(
        quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "2s"), gemini_response()
    )
    assert ask(client) == Verdict(ok=True)
    assert len(script.calls) == 2 and sleeps and sleeps[0] >= 2
    assert client.exhausted is None


def test_429_asking_to_wait_too_long_is_not_retried():
    client, script, sleeps = make_gemini(
        quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "3600s")
    )
    with pytest.raises(QuotaExhausted):
        ask(client)
    assert len(script.calls) == 1 and sleeps == []


def test_429_retry_after_header_is_used():
    error = api_error(429, "RESOURCE_EXHAUSTED", "Too many requests", headers={"retry-after": "5"})
    client, script, sleeps = make_gemini(error, gemini_response())
    assert ask(client) == Verdict(ok=True)
    assert sleeps and sleeps[0] >= 5


def test_zero_quota_model_is_exhausted_immediately():
    client, script, _ = make_gemini(
        quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", value="0")
    )
    with pytest.raises(QuotaExhausted, match="нулевая квота"):
        ask(client)
    assert len(script.calls) == 1


def test_call_budget_raises_quota_exhausted_and_stops_requests():
    client, script, _ = make_gemini(gemini_response(), max_calls=2)
    ask(client, user="1")
    ask(client, user="2")
    with pytest.raises(QuotaExhausted, match="RAVEL_MAX_CALLS"):
        ask(client, user="3")
    with pytest.raises(QuotaExhausted):
        ask(client, user="4")
    assert len(script.calls) == 2 and client.calls == 2


def test_call_budget_counts_retries():
    client, script, _ = make_gemini(api_error(503, "UNAVAILABLE", "high demand"), max_calls=2)
    with pytest.raises(QuotaExhausted):
        ask(client)
    assert len(script.calls) == 2


def test_call_budget_from_env(monkeypatch):
    monkeypatch.setenv("RAVEL_MAX_CALLS", "1")
    client, script, _ = make_gemini(gemini_response())
    assert client.max_calls == 1
    ask(client, user="1")
    with pytest.raises(QuotaExhausted):
        ask(client, user="2")
    monkeypatch.setenv("RAVEL_MAX_CALLS", "много")
    with pytest.raises(LLMUnavailable, match="RAVEL_MAX_CALLS"):
        make_gemini(gemini_response())


# ----------------------------------------------------- transient failures


def test_503_is_retried_three_times_then_unavailable():
    client, script, sleeps = make_gemini(api_error(503, "UNAVAILABLE", "This model is overloaded."))
    with pytest.raises(LLMUnavailable) as exc:
        ask(client)
    assert not isinstance(exc.value, QuotaExhausted)
    assert len(script.calls) == 3 and len(sleeps) == 2
    assert all(1 <= s <= 60 for s in sleeps)


def test_503_then_success_recovers():
    client, script, _ = make_gemini(api_error(503, "UNAVAILABLE", "overloaded"), gemini_response())
    assert ask(client) == Verdict(ok=True)
    assert len(script.calls) == 2


@pytest.mark.parametrize("module", [httpx, httpx2], ids=["httpx", "httpx2"])
@pytest.mark.parametrize("name", ["ConnectError", "RemoteProtocolError", "ReadError"])
def test_network_errors_become_unavailable_after_retries(module, name):
    client, script, _ = make_gemini(getattr(module, name)("connection refused"))
    with pytest.raises(LLMUnavailable, match="Нет связи с Gemini"):
        ask(client)
    assert len(script.calls) == 3


@pytest.mark.parametrize("module", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_read_timeout_is_retried_once_then_unavailable(module):
    client, script, _ = make_gemini(module.ReadTimeout("timed out"), timeout=30)
    with pytest.raises(LLMUnavailable, match="за 30 с"):
        ask(client)
    assert len(script.calls) == 2


def test_network_error_then_success_recovers():
    client, script, _ = make_gemini(httpx2.ConnectError("reset"), gemini_response())
    assert ask(client) == Verdict(ok=True)
    assert len(script.calls) == 2


@pytest.mark.parametrize(
    "error",
    [errors.UnknownApiResponseError("bad payload"), ValueError("boom"), httpx2.TooManyRedirects("loop")],
)
def test_any_other_sdk_exception_is_llm_error(error):
    client, script, _ = make_gemini(error)
    with pytest.raises(LLMError) as exc:
        ask(client)
    assert not isinstance(exc.value, LLMUnavailable)
    assert len(script.calls) == 1


# ------------------------------------------------- empty / truncated output


@pytest.mark.parametrize("text", ["", None, "   "], ids=["empty", "no-parts", "blank"])
def test_empty_response_retried_once_then_llm_error(text):
    client, script, _ = make_gemini(gemini_response(text))
    with pytest.raises(LLMError, match="пустой ответ") as exc:
        ask(client)
    assert not isinstance(exc.value, LLMUnavailable)
    assert len(script.calls) == 2


def test_empty_then_valid_response_recovers():
    client, script, _ = make_gemini(gemini_response(""), gemini_response())
    assert ask(client) == Verdict(ok=True)
    assert len(script.calls) == 2


def test_max_tokens_is_truncated_response_not_silent_success():
    # валидный по схеме, но обрезанный ответ не должен приниматься
    client, script, _ = make_gemini(gemini_response('{"findings": []}', "MAX_TOKENS"), max_output_tokens=512)
    with pytest.raises(TruncatedResponse, match="512"):
        ask(client, Findings)
    assert len(script.calls) == 1


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "OTHER"])
def test_other_finish_reasons_are_truncated(reason):
    client, script, _ = make_gemini(gemini_response('{"ok": true}', reason))
    with pytest.raises(TruncatedResponse, match=reason):
        ask(client)
    assert len(script.calls) == 1


def test_blocked_prompt_is_truncated_response():
    blocked = types.GenerateContentResponse(
        candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY")
    )
    client, _, _ = make_gemini(blocked)
    with pytest.raises(TruncatedResponse, match="SAFETY"):
        ask(client)


# ------------------------------------------------------------ redaction


def test_key_is_removed_from_error_text_and_text_is_capped(monkeypatch):
    other = "AIzaSyOTHER_" + "z" * 27
    monkeypatch.setenv("GEMINI_API_KEY", "env-secret-value-123")
    message = (
        f"bad key {FAKE_KEY} or {other}, url ?key={FAKE_KEY}&alt=json env-secret-value-123 " + "x" * 2000
    )
    client, _, _ = make_gemini(api_error(400, "INVALID_ARGUMENT", message))
    with pytest.raises(LLMError) as exc:
        ask(client)
    text = str(exc.value)
    assert FAKE_KEY not in text and other not in text and "AIza" not in text
    assert "env-secret-value-123" not in text
    assert "[ключ скрыт]" in text
    assert len(text) < 400


def test_provider_text_redacts_before_cutting():
    # ключ на границе обрезки не должен остаться даже частично
    text = provider_text("x" * 290 + FAKE_KEY)
    assert "AIza" not in text and len(text) <= 300


def test_llm_error_itself_redacts_keys():
    assert "AIza" not in str(LLMError(f"oops {FAKE_KEY}"))


# ---------------------------------------------------------------- ollama


def make_ollama(*outcomes, **kw) -> tuple[OllamaClient, Script]:
    client = OllamaClient("qwen-test", "http://localhost:11434", None, **kw)
    script = Script(*outcomes)
    client._client.chat = script
    client.retry_sleep = lambda s: None
    return client, script


def ollama_response(content: str = '{"ok": true}', done_reason: str = "stop"):
    return ollama.ChatResponse(
        model="qwen-test",
        message=ollama.Message(role="assistant", content=content),
        done=True,
        done_reason=done_reason,
    )


def test_ollama_passes_limits_and_timeout():
    client, script = make_ollama(ollama_response(), timeout=12, max_output_tokens=1024)
    assert ask(client) == Verdict(ok=True)
    assert script.calls[0]["options"]["num_predict"] == 1024
    assert client._client._client.timeout.read == 12


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Failed to connect to Ollama"),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
    ],
    ids=["ConnectionError", "ConnectError", "ReadTimeout"],
)
def test_ollama_unreachable_or_slow_is_unavailable_without_retries(error):
    client, script = make_ollama(error)
    with pytest.raises(LLMUnavailable):
        ask(client)
    assert len(script.calls) == 1


def test_ollama_missing_model_is_unavailable():
    client, script = make_ollama(ollama.ResponseError("model 'qwen-test' not found", 404))
    with pytest.raises(LLMUnavailable, match="ollama pull"):
        ask(client)
    assert len(script.calls) == 1


def test_ollama_500_retried_then_unavailable():
    client, script = make_ollama(ollama.ResponseError("runner crashed", 500))
    with pytest.raises(LLMUnavailable):
        ask(client)
    assert len(script.calls) == 3


def test_ollama_400_is_llm_error():
    client, _ = make_ollama(ollama.ResponseError("invalid format", 400))
    with pytest.raises(LLMError) as exc:
        ask(client)
    assert not isinstance(exc.value, LLMUnavailable)


def test_ollama_length_is_truncated():
    client, script = make_ollama(ollama_response('{"ok": true}', "length"))
    with pytest.raises(TruncatedResponse):
        ask(client)
    assert len(script.calls) == 1


def test_ollama_empty_response_retried_once():
    client, script = make_ollama(ollama_response(""))
    with pytest.raises(LLMError, match="пустой ответ"):
        ask(client)
    assert len(script.calls) == 2


# ----------------------------------------------------- base & get_client


class Scripted(LLMClient):
    """Как FakeLLM из test_agents: только _raw и super().__init__(model, cache_dir)."""

    provider = "scripted"

    def __init__(self, *outcomes, cache_dir=None):
        super().__init__("s-1", cache_dir)
        self.script = Script(*outcomes)
        self.retry_sleep = lambda s: None

    def _raw(self, system, user, schema):
        return self.script(user=user)


def test_subclass_with_legacy_init_gets_default_budget_and_cache(tmp_path):
    llm = Scripted('{"ok": true}', cache_dir=tmp_path)
    assert llm.max_calls == 50 and llm.label == "scripted/s-1"
    ask(llm)
    again = Scripted(TransientError("must not be called"), cache_dir=tmp_path)
    assert ask(again) == Verdict(ok=True)
    assert again.calls == 0 and again.cache_hits == 1


def test_transient_error_default_final_is_llm_error():
    llm = Scripted(TransientError("flaky"))
    with pytest.raises(LLMError, match="попыток: 3") as exc:
        ask(llm)
    assert not isinstance(exc.value, (LLMUnavailable, TransientError))
    assert llm.calls == 3


def test_get_client_reads_model_timeout_and_tokens(monkeypatch):
    monkeypatch.setenv("RAVEL_LLM_TIMEOUT", "30")
    monkeypatch.setenv("RAVEL_MAX_OUTPUT_TOKENS", "4096")
    client = get_client("gemini")
    assert client.model == "gemini-3.8-flash"
    assert client.timeout == 30 and client.max_output_tokens == 4096
    assert client._client._api_client._http_options.timeout == 30_000
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    assert get_client("gemini").model == "gemini-3.6-flash"


def test_get_client_rejects_bad_numbers(monkeypatch):
    monkeypatch.setenv("RAVEL_LLM_TIMEOUT", "быстро")
    with pytest.raises(LLMUnavailable, match="RAVEL_LLM_TIMEOUT"):
        get_client("gemini")
