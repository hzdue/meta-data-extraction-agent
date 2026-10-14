"""Offline provider contracts; all clients and clocks below are synthetic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, RateLimitError
from pydantic import ValidationError
import pytest

from meta_data_extraction.agent.paths import PathPolicyError, RunDirectory
from meta_data_extraction.agent.providers import (
    ChatAIProvider, LiveRunSpec, LiveSource, MockProvider, ProviderResult,
    ProviderSetupError, RequestLedger, create_live_provider,
)
import meta_data_extraction.components.llm as inherited


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (tmp_path / "src/meta_data_extraction").mkdir(parents=True)
    sources = tmp_path / "tests/agent/fixtures"
    sources.mkdir(parents=True)
    (sources / "paper.txt").write_text("Synthetic source", encoding="utf-8")
    return tmp_path


@pytest.fixture
def spec():
    return LiveRunSpec(
        endpoint="https://chatai.example.test/v1",
        sources=(LiveSource(path="tests/agent/fixtures/paper.txt", sha256=digest("Synthetic source")),),
        split="synthetic-offline-provider-test",
        profile_sha256=digest("profile"), approved_prompt_sha256=(digest("prompt"),),
        max_calls=10, max_output_tokens=100, max_total_output_tokens=1000,
        api_key_env="M02_TEST_KEY",
    )


@pytest.fixture
def clock():
    return [1000.0]


@pytest.fixture
def ledger(workspace, clock):
    return RequestLedger(workspace, clock=lambda: clock[0], monotonic_clock=lambda: clock[0])


@pytest.fixture
def ready_ledger(ledger, clock):
    clock[0] += 3600
    return ledger


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"samples": []}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
        )
        self.error = error
        self.options = {}
        self.requests = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **options):
        self.options = options
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True
        if "http_client" in self.options:
            self.options["http_client"].close()


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("M02_TEST_KEY", "synthetic-not-a-real-key")
    monkeypatch.setattr(inherited, "OpenAI", lambda **kwargs: client)
    return client


def provider(workspace, spec, ledger, *, dotenv=False):
    run = RunDirectory.create(workspace, "test-run")
    run.write_json("live-run-spec.json", spec.model_dump(mode="json"))
    return ChatAIProvider(run, spec, ledger, load_dotenv_file=dotenv)


def test_mock_is_deterministic_and_exhaustion_is_failure():
    first = ProviderResult(status="ok", content="{}")
    mock = MockProvider([first])
    assert mock.generate("one") == first
    assert mock.generate("two").error_code == "mock_exhausted"
    assert mock.requests == ("one", "two")
    assert mock.mode == "mock"
    with pytest.raises(ValueError):
        MockProvider([ProviderResult(status="ok", mode="live")])


def test_result_is_frozen_and_unknown_usage_stays_unknown():
    result = ProviderResult(status="ok", content="{}")
    assert result.input_tokens is None and result.output_tokens is None
    with pytest.raises(ValidationError):
        result.content = "changed"
    with pytest.raises(ValidationError):
        ProviderResult(status="ok", input_tokens=True)


@pytest.mark.parametrize("update", [
    {"model": "unapproved-model"}, {"endpoint": "http://host/v1"},
    {"endpoint": "https://user:secret@host/v1"}, {"endpoint": "https://host/v1?api_key=secret"},
    {"max_calls": 11}, {"max_calls": True}, {"max_output_tokens": 0},
    {"max_total_output_tokens": 99}, {"timeout_seconds": 301},
    {"profile_sha256": "not-a-hash"}, {"approved_prompt_sha256": ()},
    {"api_key_env": "bad-key"}, {"sources": ()},
])
def test_live_spec_rejects_unsafe_or_incomplete_scope(spec, update):
    with pytest.raises(ValidationError):
        LiveRunSpec.model_validate(spec.model_dump() | update)


def test_new_ledger_fails_closed_until_full_hour_and_survives_restart(workspace, ledger, spec, clock):
    assert ledger.reserve("first", spec).error_code == "request_history_unknown"
    clock[0] += 3599.9
    assert RequestLedger(workspace, clock=lambda: clock[0], monotonic_clock=lambda: clock[0]).reserve("first", spec).error_code == "request_history_unknown"
    clock[0] += 0.1
    assert RequestLedger(workspace, clock=lambda: clock[0], monotonic_clock=lambda: clock[0]).reserve("first", spec).request_id


def test_ten_attempts_shared_across_runs_and_exact_rolling_window(workspace, ready_ledger, spec, clock):
    for index in range(10):
        reservation = ready_ledger.reserve(f"run-{index}", spec)
        assert reservation.request_id
        assert ready_ledger.complete(ProviderResult(status="failed", request_id=reservation.request_id, mode="live"))
    assert RequestLedger(workspace, clock=lambda: clock[0], monotonic_clock=lambda: clock[0]).reserve("eleventh", spec).error_code == "rolling_request_limit"
    clock[0] += 3599
    assert ready_ledger.reserve("eleventh", spec).error_code == "rolling_request_limit"
    clock[0] += 1
    assert ready_ledger.reserve("eleventh", spec).request_id


def test_concurrent_threads_cannot_exceed_global_limit(ready_ledger, spec):
    with ThreadPoolExecutor(max_workers=12) as workers:
        results = list(workers.map(lambda index: ready_ledger.reserve(f"thread-{index}", spec), range(24)))
    assert sum(result.request_id is not None for result in results) == 10
    assert sum(result.error_code == "rolling_request_limit" for result in results) == 14


def test_concurrent_processes_share_durable_reservations(workspace, ready_ledger, spec, clock):
    code = (
        "import json,sys; from pathlib import Path; "
        "from meta_data_extraction.agent.providers import LiveRunSpec,RequestLedger; "
        "spec=LiveRunSpec.model_validate_json(sys.argv[2]); "
        "ledger=RequestLedger(Path(sys.argv[1]),clock=lambda:4600.0,monotonic_clock=lambda:4600.0); "
        "print(sum(ledger.reserve(sys.argv[3]+'-'+str(i),spec).request_id is not None for i in range(4)))"
    )
    children = [subprocess.Popen(
        [sys.executable, "-c", code, str(workspace), spec.model_dump_json(), f"process-{index}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy(),
    ) for index in range(4)]
    total = 0
    for child in children:
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        total += int(stdout.strip())
    assert total == 10
    assert ready_ledger.reserve("extra", spec).error_code == "rolling_request_limit"


def test_readiness_does_not_spend_attempts_and_scope_is_frozen(ready_ledger, spec):
    assert ready_ledger.reserve("run", spec, commit_attempt=False).request_id is None
    for _ in range(10):
        assert ready_ledger.reserve("run", spec).request_id
    modified = spec.model_copy(update={"max_output_tokens": 99})
    assert ready_ledger.reserve("run", modified).error_code == "run_scope_changed"


def test_run_request_and_output_budgets_are_conservative(ready_ledger, spec):
    calls = spec.model_copy(update={"max_calls": 1})
    assert ready_ledger.reserve("calls", calls).request_id
    assert ready_ledger.reserve("calls", calls).error_code == "run_request_budget"
    tokens = spec.model_copy(update={"max_total_output_tokens": 150})
    assert ready_ledger.reserve("tokens", tokens).request_id
    assert ready_ledger.reserve("tokens", tokens).error_code == "run_output_budget"


def test_clock_rollback_fails_closed(ready_ledger, spec, clock):
    assert ready_ledger.reserve("one", spec).request_id
    clock[0] -= 1
    assert ready_ledger.reserve("two", spec).error_code == "ledger_clock_regressed"


def test_corrupt_or_empty_existing_ledger_does_not_reset_history(workspace):
    path = workspace / RequestLedger.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    with pytest.raises(ProviderSetupError, match="ledger_unavailable"):
        RequestLedger(workspace)
    path.write_bytes(b"not a sqlite database")
    with pytest.raises(ProviderSetupError, match="ledger_unavailable"):
        RequestLedger(workspace)


def test_ledger_hardlink_to_released_output_is_rejected(workspace):
    target = workspace / "released.sqlite3"
    target.write_bytes(b"protected")
    path = workspace / RequestLedger.relative_path
    path.parent.mkdir(parents=True)
    os.link(target, path)
    with pytest.raises(PathPolicyError):
        RequestLedger(workspace)
    assert target.read_bytes() == b"protected"


def test_factory_freezes_scope_without_credentials_or_network(workspace, spec, monkeypatch):
    monkeypatch.delenv("M02_TEST_KEY", raising=False)
    monkeypatch.setattr(inherited, "OpenAI", lambda **kw: pytest.fail("No client before dispatch"))
    run = RunDirectory.create(workspace, "factory")
    live = create_live_provider(run, spec, load_dotenv_file=True)
    assert live.model_name == "qwen3.8-27b" and live.mode == "live"
    assert create_live_provider(run, spec).spec == spec
    assert json.loads((run.path / "live-run-spec.json").read_text())["max_calls"] == 10
    assert live.generate("prompt").error_code == "request_history_unknown"
    changed = spec.model_copy(update={"max_calls": 9})
    with pytest.raises(ProviderSetupError, match="run_scope_changed"):
        create_live_provider(run, changed)


def test_factory_rejects_hash_mismatch_and_offline_source(workspace, spec):
    run = RunDirectory.create(workspace, "bad-source")
    bad = spec.model_copy(update={"sources": (LiveSource(path="tests/agent/fixtures/paper.txt", sha256="0" * 64),)})
    with pytest.raises(ProviderSetupError, match="source_hash_mismatch"):
        create_live_provider(run, bad)
    bad = spec.model_copy(update={"sources": (LiveSource(path="data/gold/paper.json", sha256="0" * 64),)})
    with pytest.raises(PathPolicyError):
        create_live_provider(run, bad)


def test_missing_credentials_is_explicit_and_does_not_reserve(workspace, spec, ready_ledger, monkeypatch):
    monkeypatch.delenv("M02_TEST_KEY", raising=False)
    live = provider(workspace, spec, ready_ledger)
    result = live.generate("prompt")
    assert result.error_code == "missing_credentials"
    assert result.request_id is None
    assert ready_ledger.reserve("test-run", spec).request_id


def test_prompt_allowlist_and_size_checked_before_sdk(workspace, spec, ready_ledger, monkeypatch):
    monkeypatch.setattr(inherited, "OpenAI", lambda **kw: pytest.fail("Blocked prompt must not construct SDK"))
    live = provider(workspace, spec, ready_ledger)
    assert live.generate("different").error_code == "prompt_not_approved"
    assert live.generate("x" * 100001).error_code == "prompt_size_limit"


def test_inherited_adapter_reused_with_no_retries_or_redirects(workspace, spec, ready_ledger, fake_client):
    live = provider(workspace, spec, ready_ledger)
    result = live.generate("prompt")
    assert result.status == "ok" and result.mode == "live"
    assert result.content == '{"samples": []}' and result.finish_reason == "stop"
    assert (result.input_tokens, result.output_tokens) == (12, 5)
    assert result.request_id and result.latency_ms is not None
    assert fake_client.options["max_retries"] == 0
    assert fake_client.options["timeout"] == 60.0
    assert fake_client.options["http_client"].follow_redirects is False
    assert fake_client.requests == [{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "prompt"}], "max_tokens": 100, "temperature": 0.0}]
    assert fake_client.closed


def test_missing_usage_stays_unknown_and_truncation_is_preserved(workspace, spec, ready_ledger, fake_client):
    fake_client.response.usage = None
    fake_client.response.choices[0].finish_reason = "length"
    result = provider(workspace, spec, ready_ledger).generate("prompt")
    assert result.input_tokens is None and result.output_tokens is None
    assert result.finish_reason == "length"


@pytest.mark.parametrize("error,status,code", [
    (APITimeoutError(request=httpx.Request("POST", "https://synthetic.test")), "timeout", "provider_timeout"),
    (RateLimitError("synthetic secret must not be logged", response=httpx.Response(429, request=httpx.Request("POST", "https://synthetic.test")), body=None), "rate_limited", "provider_429"),
    (RuntimeError("synthetic secret must not be logged"), "failed", "provider_failure"),
])
def test_failed_attempts_are_counted_once_with_safe_errors(workspace, spec, ready_ledger, fake_client, error, status, code):
    fake_client.error = error
    bounded = spec.model_copy(update={"max_calls": 1})
    live = provider(workspace, bounded, ready_ledger)
    result = live.generate("prompt")
    assert result.status == status and result.error_code == code and result.request_id
    assert "synthetic secret" not in result.model_dump_json()
    assert len(fake_client.requests) == 1
    assert live.generate("prompt").error_code == "run_request_budget"
    with sqlite3.connect(workspace / RequestLedger.relative_path) as connection:
        row = connection.execute("SELECT status, input_tokens, output_tokens, error_code FROM attempts").fetchone()
        assert row == (status, None, None, code)


def test_process_interruption_reservation_is_never_refunded(workspace, ready_ledger, spec, clock):
    bounded = spec.model_copy(update={"max_calls": 1})
    reservation = ready_ledger.reserve("interrupted", bounded)
    assert reservation.request_id
    restarted = RequestLedger(workspace, clock=lambda: clock[0], monotonic_clock=lambda: clock[0])
    assert restarted.reserve("interrupted", bounded).error_code == "run_request_budget"


def test_ledger_never_stores_prompt_or_credentials(workspace, spec, ready_ledger, fake_client):
    provider(workspace, spec, ready_ledger).generate("prompt")
    raw = (workspace / RequestLedger.relative_path).read_bytes()
    assert b"synthetic-not-a-real-key" not in raw
    assert b"Synthetic source" not in raw
    with sqlite3.connect(workspace / RequestLedger.relative_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_mock_configuration_fingerprint_binds_initial_responses():
    result = ProviderResult(status="ok", content="{}")
    mock = MockProvider([result])
    fingerprint = mock.configuration_fingerprint
    mock.generate("one")
    assert mock.configuration_fingerprint == fingerprint
    assert fingerprint == MockProvider([result]).configuration_fingerprint
    assert fingerprint != MockProvider([result], model_name="other").configuration_fingerprint
    assert fingerprint != MockProvider([ProviderResult(status="ok", content="[]")]).configuration_fingerprint


def test_scope_and_source_changes_block_before_sdk(workspace, spec, ready_ledger, monkeypatch):
    monkeypatch.setattr(inherited, "OpenAI", lambda **kw: pytest.fail("Changed scope must not construct SDK"))
    live = provider(workspace, spec, ready_ledger)
    source = workspace / "tests/agent/fixtures/paper.txt"
    source.write_text("edited source", encoding="utf-8")
    assert live.generate("prompt").error_code == "source_hash_mismatch"
    source.write_text("Synthetic source", encoding="utf-8")
    manifest = workspace / "runs/agent/test-run/live-run-spec.json"
    manifest.write_text(spec.model_copy(update={"max_calls": 9}).model_dump_json(), encoding="utf-8")
    assert live.generate("prompt").error_code == "run_scope_changed"
    manifest.unlink()
    assert live.generate("prompt").error_code == "run_scope_unavailable"


def test_first_hour_never_loads_dotenv(workspace, spec, ledger, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: pytest.fail("Unknown history must not read .env"))
    live = provider(workspace, spec, ledger, dotenv=True)
    assert live.generate("prompt").error_code == "request_history_unknown"


def test_ledger_write_failure_preserves_raw_output_and_attempt_id(workspace, spec, ready_ledger, fake_client, monkeypatch):
    monkeypatch.setattr(ready_ledger, "complete", lambda result: False)
    result = provider(workspace, spec, ready_ledger).generate("prompt")
    assert result.status == "failed" and result.error_code == "ledger_record_failed"
    assert result.request_id and result.content == '{"samples": []}'


def test_unfinished_reservations_never_age_out(workspace, ready_ledger, spec, clock):
    reservations = [ready_ledger.reserve(f"pending-{index}", spec) for index in range(10)]
    clock[0] += 7200
    assert ready_ledger.reserve("late", spec).error_code == "rolling_request_limit"
    for reservation in reservations:
        assert ready_ledger.complete(ProviderResult(status="failed", request_id=reservation.request_id, mode="live"))
    clock[0] += 3599
    assert ready_ledger.reserve("late", spec).error_code == "rolling_request_limit"
    clock[0] += 1
    assert ready_ledger.reserve("late", spec).request_id


@pytest.mark.parametrize("timestamp", ["broken", -1.0, float("inf")])
def test_invalid_metadata_timestamps_fail_closed(workspace, ready_ledger, spec, clock, timestamp):
    with sqlite3.connect(workspace / RequestLedger.relative_path) as connection:
        connection.execute("UPDATE meta SET last_seen=? WHERE id=1", (timestamp,))
    assert ready_ledger.reserve("corrupted", spec).error_code == "ledger_unavailable"
    with pytest.raises(ProviderSetupError, match="ledger_unavailable"):
        RequestLedger(workspace, clock=lambda: clock[0], monotonic_clock=lambda: clock[0])


def test_wall_clock_forward_jump_cannot_grant_early_allowance(workspace, spec):
    wall, monotonic = [1000.0], [10.0]
    ledger = RequestLedger(workspace, clock=lambda: wall[0], monotonic_clock=lambda: monotonic[0])
    wall[0] += 7200
    assert ledger.reserve("warmup", spec).error_code == "request_history_unknown"
    monotonic[0] += 3600
    for index in range(10):
        reservation = ledger.reserve(f"clock-{index}", spec)
        assert reservation.request_id
        assert ledger.complete(ProviderResult(status="failed", request_id=reservation.request_id, mode="live"))
    wall[0] += 7200
    assert ledger.reserve("too-early", spec).error_code == "rolling_request_limit"
    monotonic[0] += 3599
    assert ledger.reserve("too-early", spec).error_code == "rolling_request_limit"
    monotonic[0] += 1
    assert ledger.reserve("aged-out", spec).request_id


def test_monotonic_epoch_regression_fails_closed(workspace, spec):
    wall, monotonic = [1000.0], [1000.0]
    ledger = RequestLedger(workspace, clock=lambda: wall[0], monotonic_clock=lambda: monotonic[0])
    wall[0] += 3600
    monotonic[0] += 3600
    assert ledger.reserve("before-reboot", spec).request_id
    wall[0] += 7200
    monotonic[0] = 50.0
    restarted = RequestLedger(workspace, clock=lambda: wall[0], monotonic_clock=lambda: monotonic[0])
    assert restarted.reserve("after-reboot", spec).error_code == "ledger_clock_regressed"


@pytest.mark.parametrize("case,expected_status,error_code", [
    ("rate_limit", "rate_limited", "provider_429"),
    ("redirect", "failed", "provider_http_error"),
    ("timeout", "timeout", "provider_timeout"),
    ("success", "ok", None),
])
def test_installed_sdk_transport_makes_exactly_one_attempt(
    workspace, spec, ready_ledger, monkeypatch, case, expected_status, error_code,
):
    """Exercise the real installed SDK/adapter; MockTransport never opens sockets."""
    import openai

    requests = []
    client_options = []
    clients = []
    original_client_factory = openai.DefaultHttpxClient

    def handle_request(request):
        requests.append(request)
        assert str(request.url) == "https://chatai.example.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-sdk-transport-key"
        assert json.loads(request.content)["model"] == "qwen3.8-27b"
        if case == "rate_limit":
            return httpx.Response(429, headers={"retry-after": "0"}, json={
                "error": {"message": "synthetic rate limit", "type": "rate_limit_error"},
            })
        if case == "redirect":
            return httpx.Response(307, headers={"location": "https://unapproved.example.test/redirected"})
        if case == "timeout":
            raise httpx.ReadTimeout("synthetic transport timeout", request=request)
        # Missing usage is deliberate: use the SDK's actual optional-field model.
        return httpx.Response(200, json={
            "id": "synthetic-completion", "object": "chat.completion", "created": 1000,
            "model": "qwen3.8-27b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
        })

    def transport_client_factory(**options):
        client_options.append(options)
        client = original_client_factory(transport=httpx.MockTransport(handle_request), **options)
        clients.append(client)
        return client

    monkeypatch.setenv("M02_TEST_KEY", "synthetic-sdk-transport-key")
    # Keep OpenAI, its HTTP request handling, retry logic, and response parsing real.
    monkeypatch.setattr(openai, "DefaultHttpxClient", transport_client_factory)
    result = provider(workspace, spec, ready_ledger).generate("prompt")

    assert result.status == expected_status and result.error_code == error_code
    assert result.request_id is not None
    assert len(requests) == 1  # retries and redirects would invoke MockTransport again
    assert client_options == [{"follow_redirects": False, "timeout": 60.0}]
    assert clients[0].is_closed
    assert requests[0].extensions["timeout"] == {
        "connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0,
    }
    assert result.input_tokens is None and result.output_tokens is None
    if case == "success":
        assert result.content == "{}" and result.finish_reason == "stop"
    else:
        assert result.content is None
    with sqlite3.connect(workspace / RequestLedger.relative_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        assert connection.execute("SELECT status FROM attempts").fetchone()[0] == expected_status
