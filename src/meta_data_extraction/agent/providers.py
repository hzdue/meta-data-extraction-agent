"""Typed providers and conservative, process-shared ChatAI request accounting.

The inherited client's request builder and response adapter are reused unchanged.
All live attempts must pass the same checkout-local SQLite ledger.  A newly
created ledger waits a full rolling hour because earlier service usage is unknown.
This is a local application boundary: other programs using the same API account
must share this ledger too; deleting it never grants immediate fresh allowance.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Literal, Protocol
from urllib.parse import urlsplit
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from meta_data_extraction.agent.paths import RunDirectory, checked_path, input_file, validate_run_id
from meta_data_extraction.components.llm import OpenAICompatibleLLM


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Literal["ok", "timeout", "rate_limited", "failed", "blocked"]
    content: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    error_code: str | None = None
    request_id: str | None = None
    mode: Literal["mock", "live"] = "mock"


class Provider(Protocol):
    model_name: str
    mode: Literal["mock", "live"]

    def generate(self, prompt: str) -> ProviderResult: ...


class MockProvider:
    """Deterministic queued responses; exhaustion is an explicit failed call."""

    mode: Literal["mock"] = "mock"

    def __init__(self, results: Sequence[ProviderResult], *, model_name: str = "mock"):
        if any(result.mode != "mock" for result in results):
            raise ValueError("Mock responses must be marked mock")
        self.model_name = model_name
        self._results = deque(results)
        configuration = {"model_name": model_name, "results": [result.model_dump(mode="json") for result in results]}
        self._configuration_fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self._requests: list[str] = []

    @property
    def configuration_fingerprint(self) -> str:
        return self._configuration_fingerprint

    @property
    def requests(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def generate(self, prompt: str) -> ProviderResult:
        self._requests.append(prompt)
        if not self._results:
            return ProviderResult(status="failed", error_code="mock_exhausted")
        return self._results.popleft()


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


class LiveSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    sha256: str

    _valid_sha256 = field_validator("sha256")(_sha256)


class LiveRunSpec(BaseModel):
    """Explicit scope frozen before use; no key or document content is recorded.

    max_total_output_tokens reserves the full requested output capacity for every
    attempt, including failures. Input usage remains measured or unknown; the
    separate byte limit bounds prompt size without pretending to know tokenization.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    endpoint: str
    model: Literal["qwen3.8-27b"] = "qwen3.8-27b"
    sources: tuple[LiveSource, ...] = Field(min_length=1)
    split: str = Field(min_length=1, max_length=120)
    profile_sha256: str
    approved_prompt_sha256: tuple[str, ...] = Field(min_length=1)
    max_calls: int = Field(ge=1, le=10, strict=True)
    max_output_tokens: int = Field(ge=1, le=32768, strict=True)
    max_total_output_tokens: int = Field(ge=1, strict=True)
    max_prompt_bytes: int = Field(default=100_000, ge=1, le=1_000_000, strict=True)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)
    api_key_env: str = "CHATAI_API_KEY"

    _valid_profile = field_validator("profile_sha256")(_sha256)

    @field_validator("endpoint")
    @classmethod
    def _endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (any(character.isspace() for character in value) or parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("Expected an explicit HTTPS endpoint without credentials or query")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _key_env(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", value):
            raise ValueError("Expected an environment variable name")
        return value

    @field_validator("approved_prompt_sha256")
    @classmethod
    def _prompts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("Prompt fingerprints must be distinct")
        return tuple(_sha256(value) for value in values)

    @model_validator(mode="after")
    def _scope(self) -> LiveRunSpec:
        if len({source.path for source in self.sources}) != len(self.sources):
            raise ValueError("Source paths must be distinct")
        if self.max_total_output_tokens < self.max_output_tokens:
            raise ValueError("Output budget must cover at least one reserved request")
        return self

    @property
    def fingerprint(self) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProviderSetupError(ValueError):
    """Setup failure with a safe code, never a raw SDK exception or credential."""

    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(error_code)


class Reservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str | None = None
    error_code: str | None = None


class RequestLedger:
    """Atomic shared reservations with a hard ten-attempt rolling-hour limit.

    Wall and same-host monotonic time must both age out. Clock rollback (including
    an uncertain monotonic epoch after reboot) fails closed; version-1 ledgers
    are not silently migrated or reset. Such cases need explicit reconciliation.
    """

    relative_path = "runs/agent/_control/chatai-requests.sqlite3"
    window_seconds = 3600.0
    request_limit = 10

    def __init__(self, root: Path | str, *, clock: Callable[[], float] = time.time,
                 monotonic_clock: Callable[[], float] = time.monotonic):
        self.root = Path(root)
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = self._path().exists()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
                if not tables:
                    if existed:
                        raise ProviderSetupError("ledger_unavailable")
                    now, monotonic_now = self._now(), self._now(monotonic=True)
                    connection.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY, version INTEGER NOT NULL, created_at REAL NOT NULL, created_monotonic REAL NOT NULL, last_seen REAL NOT NULL, last_monotonic REAL NOT NULL)")
                    connection.execute("INSERT INTO meta VALUES (1, 2, ?, ?, ?, ?)", (now, monotonic_now, now, monotonic_now))
                    connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, spec_sha256 TEXT NOT NULL)")
                    connection.execute("CREATE TABLE attempts (request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, attempted_at REAL NOT NULL, attempted_monotonic REAL NOT NULL, completed_at REAL, completed_monotonic REAL, reserved_output_tokens INTEGER NOT NULL, status TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL, error_code TEXT)")
                    connection.execute("CREATE INDEX attempts_time ON attempts (attempted_at)")
                self._metadata(connection)
        except sqlite3.Error:
            raise ProviderSetupError("ledger_unavailable") from None

    def _path(self) -> Path:
        path = checked_path(self.root, self.relative_path, allowed_roots=("runs/agent/_control",))
        # SQLite may open auxiliary files itself; validate these names too.
        for suffix in ("-journal", "-wal", "-shm"):
            checked_path(self.root, self.relative_path + suffix, allowed_roots=("runs/agent/_control",))
        return path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path(), timeout=5.0)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _now(self, *, monotonic: bool = False) -> float:
        value = self._monotonic_clock() if monotonic else self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
            raise ProviderSetupError("ledger_clock_invalid")
        return float(value)

    @staticmethod
    def _metadata(connection: sqlite3.Connection) -> tuple[float, float, float, float]:
        rows = connection.execute("SELECT version, created_at, created_monotonic, last_seen, last_monotonic FROM meta WHERE id=1").fetchall()
        if len(rows) != 1 or rows[0][0] != 2:
            raise ProviderSetupError("ledger_unavailable")
        created_at, created_monotonic, last_seen, last_monotonic = rows[0][1:]
        if (any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in (created_at, created_monotonic, last_seen, last_monotonic))
                or created_at > last_seen or created_monotonic > last_monotonic):
            raise ProviderSetupError("ledger_unavailable")
        return tuple(float(value) for value in rows[0][1:])

    def reserve(self, run_id: str, spec: LiveRunSpec, *, commit_attempt: bool = True) -> Reservation:
        """Readiness checks do not spend an attempt; dispatch reservations always do.

        A reserved attempt is never refunded after failures or process interruption.
        Unfinished reservations occupy allowance indefinitely; finished requests
        occupy it for a full hour after completion, conservatively covering any
        process suspension between reservation and the actual network dispatch.
        """
        validate_run_id(run_id)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                created_at, created_monotonic, last_seen, last_monotonic = self._metadata(connection)
                now, monotonic_now = self._now(), self._now(monotonic=True)
                if now < last_seen or monotonic_now < last_monotonic:
                    return Reservation(error_code="ledger_clock_regressed")
                connection.execute("UPDATE meta SET last_seen=?, last_monotonic=? WHERE id=1", (now, monotonic_now))
                if min(now - created_at, monotonic_now - created_monotonic) < self.window_seconds:
                    return Reservation(error_code="request_history_unknown")
                row = connection.execute("SELECT spec_sha256 FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if row and row[0] != spec.fingerprint:
                    return Reservation(error_code="run_scope_changed")
                count = connection.execute("SELECT COUNT(*) FROM attempts WHERE completed_at IS NULL OR completed_monotonic IS NULL OR completed_at > ? OR completed_monotonic > ?", (now - self.window_seconds, monotonic_now - self.window_seconds)).fetchone()[0]
                if count >= self.request_limit:
                    return Reservation(error_code="rolling_request_limit")
                calls, tokens = connection.execute("SELECT COUNT(*), COALESCE(SUM(reserved_output_tokens), 0) FROM attempts WHERE run_id=?", (run_id,)).fetchone()
                if calls >= spec.max_calls:
                    return Reservation(error_code="run_request_budget")
                if tokens + spec.max_output_tokens > spec.max_total_output_tokens:
                    return Reservation(error_code="run_output_budget")
                if not commit_attempt:
                    return Reservation()
                connection.execute("INSERT OR IGNORE INTO runs VALUES (?, ?)", (run_id, spec.fingerprint))
                request_id = uuid.uuid4().hex
                connection.execute("INSERT INTO attempts (request_id,run_id,attempted_at,attempted_monotonic,reserved_output_tokens,status) VALUES (?,?,?,?,?,?)", (request_id, run_id, now, monotonic_now, spec.max_output_tokens, "reserved"))
                return Reservation(request_id=request_id)
        except (sqlite3.Error, ValueError, OSError):
            return Reservation(error_code="ledger_unavailable")

    def complete(self, result: ProviderResult) -> bool:
        """Attach safe observable outcomes; content and credentials never enter SQL."""
        if result.request_id is None:
            return False
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _, _, last_seen, last_monotonic = self._metadata(connection)
                now, monotonic_now = self._now(), self._now(monotonic=True)
                if now < last_seen or monotonic_now < last_monotonic:
                    return False
                connection.execute("UPDATE meta SET last_seen=?, last_monotonic=? WHERE id=1", (now, monotonic_now))
                changed = connection.execute("UPDATE attempts SET status=?, completed_at=?, completed_monotonic=?, input_tokens=?, output_tokens=?, latency_ms=?, error_code=? WHERE request_id=? AND status='reserved'", (result.status, now, monotonic_now, result.input_tokens, result.output_tokens, result.latency_ms, result.error_code, result.request_id)).rowcount
            return changed == 1
        except (sqlite3.Error, ValueError, OSError):
            return False


class ChatAIProvider:
    """One request per generate call; scope and reservations precede dispatch."""

    mode: Literal["live"] = "live"

    def __init__(self, run: RunDirectory, spec: LiveRunSpec, ledger: RequestLedger,
                 *, load_dotenv_file: bool = False):
        if run.root.resolve() != ledger.root.resolve():
            raise ProviderSetupError("ledger_workspace_mismatch")
        self._run = run
        self._spec = LiveRunSpec.model_validate(spec.model_dump(mode="json"))
        self._ledger = ledger
        self._load_dotenv_file = load_dotenv_file

    @property
    def spec(self) -> LiveRunSpec:
        return self._spec

    @property
    def model_name(self) -> str:
        return self.spec.model

    def _blocked(self, code: str) -> ProviderResult:
        status = "rate_limited" if code == "rolling_request_limit" else "blocked"
        return ProviderResult(status=status, error_code=code, mode="live")

    def generate(self, prompt: str) -> ProviderResult:
        try:
            manifest = checked_path(self._run.root, f"runs/agent/{self._run.run_id}/live-run-spec.json", allowed_roots=("runs/agent",))
            if LiveRunSpec.model_validate_json(manifest.read_text(encoding="utf-8")) != self.spec:
                return self._blocked("run_scope_changed")
            for source in self.spec.sources:
                path = input_file(self._run.root, source.path, allowed_roots=("data/raw", "tests/agent/fixtures"))
                if hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
                    return self._blocked("source_hash_mismatch")
        except (ValueError, OSError):
            return self._blocked("run_scope_unavailable")
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > self.spec.max_prompt_bytes:
            return self._blocked("prompt_size_limit")
        if hashlib.sha256(prompt_bytes).hexdigest() not in self.spec.approved_prompt_sha256:
            return self._blocked("prompt_not_approved")
        readiness = self._ledger.reserve(self._run.run_id, self.spec, commit_attempt=False)
        if readiness.error_code:
            return self._blocked(readiness.error_code)
        if self._load_dotenv_file:
            try:
                from dotenv import load_dotenv
                env_path = checked_path(self._run.root, ".env", allowed_roots=(".env",))
                load_dotenv(env_path, override=False)
            except (ValueError, OSError):
                return self._blocked("credentials_unavailable")
        if not os.environ.get(self.spec.api_key_env):
            return self._blocked("missing_credentials")
        try:
            from openai import DefaultHttpxClient
            adapter = OpenAICompatibleLLM(
                self.spec.model, api_key_env=self.spec.api_key_env,
                base_url=self.spec.endpoint, max_tokens=self.spec.max_output_tokens,
            )
            original_client = adapter._client
            adapter._client = original_client.with_options(
                max_retries=0, timeout=self.spec.timeout_seconds,
                http_client=DefaultHttpxClient(follow_redirects=False, timeout=self.spec.timeout_seconds),
            )
            if adapter._client is not original_client:
                original_client.close()
        except Exception:
            return self._blocked("client_setup_failed")
        reservation = self._ledger.reserve(self._run.run_id, self.spec)
        if reservation.error_code:
            try:
                adapter._client.close()
            except Exception:
                pass
            return self._blocked(reservation.error_code)
        start = time.perf_counter()
        try:
            response = adapter.generate(prompt)
            result = ProviderResult(
                status="ok", content=response.content, finish_reason=response.finish_reason,
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                latency_ms=(time.perf_counter() - start) * 1000,
                request_id=reservation.request_id, mode="live",
            )
        except Exception as error:
            from openai import APITimeoutError, RateLimitError, APIStatusError
            if isinstance(error, (APITimeoutError, TimeoutError)):
                status, code = "timeout", "provider_timeout"
            elif isinstance(error, RateLimitError):
                status, code = "rate_limited", "provider_429"
            elif isinstance(error, APIStatusError):
                status, code = "failed", "provider_http_error"
            else:
                status, code = "failed", "provider_failure"
            result = ProviderResult(status=status, error_code=code,
                latency_ms=(time.perf_counter() - start) * 1000,
                request_id=reservation.request_id, mode="live")
        finally:
            try:
                adapter._client.close()
            except Exception:
                pass
        if not self._ledger.complete(result):
            return result.model_copy(update={"status": "failed", "error_code": "ledger_record_failed"})
        return result


def create_live_provider(run: RunDirectory, spec: LiveRunSpec, *,
                         load_dotenv_file: bool = False) -> ChatAIProvider:
    """Persist declared scope without reading credentials or making any request.

    The caller must verify the actual extraction profile and rendered prompt
    against this spec. Sources are independently checked here against actual bytes.
    Reopening an identical spec is safe; changing its run scope is rejected.
    """
    spec = LiveRunSpec.model_validate(spec.model_dump(mode="json"))
    for source in spec.sources:
        path = input_file(run.root, source.path, allowed_roots=("data/raw", "tests/agent/fixtures"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
            raise ProviderSetupError("source_hash_mismatch")
    manifest = checked_path(run.root, f"runs/agent/{run.run_id}/live-run-spec.json", allowed_roots=("runs/agent",))
    if manifest.exists():
        try:
            recorded = LiveRunSpec.model_validate_json(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise ProviderSetupError("run_scope_unreadable") from None
        if recorded != spec:
            raise ProviderSetupError("run_scope_changed")
    else:
        run.write_json("live-run-spec.json", spec.model_dump(mode="json"))
    ledger = RequestLedger(run.root)
    return ChatAIProvider(run, spec, ledger, load_dotenv_file=load_dotenv_file)
