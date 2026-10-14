"""Actual ChatAI adapter/ledger with invented SDK responses and clocks, no network."""
from collections import deque
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import pytest

from test_providers import FakeClient
from meta_data_extraction.agent import providers
from meta_data_extraction.agent.pilot_bundle import BUNDLE_FILES
from meta_data_extraction.agent.publication_run import run_publication, _child, _matches_field
from meta_data_extraction.agent.publication_types import PublicationSpec, load_publication_spec
from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.components import llm as inherited


@pytest.fixture
def live_case(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("l")
    repository = Path(__file__).resolve().parents[2]
    files = set(BUNDLE_FILES) | {"configs/agent/publication_synthetic.json"}
    files.update(p.relative_to(repository).as_posix() for p in
                 (repository/"src/meta_data_extraction").rglob("*.py"))
    files.update(p.relative_to(repository).as_posix() for p in
                 (repository/"tests/agent/fixtures").glob("publication_*"))
    for relative in files:
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository/relative, target)
    original = load_publication_spec(root, "configs/agent/publication_synthetic.json")
    responses = [(root/pin.path).read_text(encoding="utf-8") for pin in original.mock_responses]
    spec = PublicationSpec.model_validate(original.model_dump(mode="json") | {
        "run_id":"live-test", "mode":"live", "mock_responses":[],
        "live":{"endpoint":"https://configured.example.invalid/v1", "api_key_env":"PUBLICATION_TEST_KEY"},
    })
    clock = [1000.0]
    real_ledger = providers.RequestLedger
    def ledger_factory(selected_root):
        assert Path(selected_root) == root
        return real_ledger(root, clock=lambda:clock[0], monotonic_clock=lambda:clock[0])
    ledger_factory(root)
    clock[0] += 3600
    monkeypatch.setattr(providers, "RequestLedger", ledger_factory)
    monkeypatch.setenv("PUBLICATION_TEST_KEY", "synthetic-test-credential")
    queue, clients = deque(), []
    def client_factory(**kwargs):
        assert kwargs["base_url"] == spec.live.endpoint
        if not queue:
            pytest.fail("Unexpected additional SDK call")
        item = queue.popleft()
        if isinstance(item, Exception):
            client = FakeClient(error=item)
        else:
            content, usage = item if isinstance(item, tuple) else (
                item, SimpleNamespace(prompt_tokens=30, completion_tokens=20))
            client = FakeClient(response=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                usage=usage))
        clients.append(client)
        return client
    monkeypatch.setattr(inherited, "OpenAI", client_factory)
    return root, spec, responses, queue, clients, clock


def execute(case, **changes):
    root, spec, _, _, _, _ = case
    selected = PublicationSpec.model_validate(spec.model_dump(mode="json") | changes)
    return run_publication(root, selected, configured_chatai_endpoint=selected.live.endpoint)


@pytest.mark.parametrize("policy", ["fixed", "issue_aware"])
def test_all_five_stages_use_exact_pinned_prompts_and_one_shared_ledger(live_case, policy):
    root, spec, responses, queue, clients, _ = live_case
    queue.extend(responses)
    result = execute(live_case, policy=policy)
    assert result["status"] == "pending_review"
    assert result["new_live_requests"] == result["total_model_calls"] == 5
    assert result["known_input_tokens"] == 150 and result["known_output_tokens"] == 100
    assert result["usage_unknown"] is False and result["cost"] is None
    ledger_path = root/"runs/agent/_control/chatai-requests.sqlite3"
    with sqlite3.connect(ledger_path) as db:
        rows = db.execute("SELECT run_id,status FROM attempts").fetchall()
    assert len(rows) == 5 and all(status == "ok" for _,status in rows)
    scopes = {}
    for stage in ("extract","check","proposal","provider"):
        scopes[stage] = json.loads((root/f"runs/agent/{_child(spec,stage)}/live-run-spec.json").read_bytes())
    assert sorted(row[0] for row in rows) == sorted(
        _child(spec,stage) for stage in ("extract","check","proposal","provider","provider"))
    for client, stage in zip(clients, ("extract","check","proposal","provider","provider"), strict=True):
        assert client.closed and client.options["max_retries"] == 0
        assert len(client.requests) == 1
        prompt = client.requests[0]["messages"][-1]["content"]
        assert sha256(prompt.encode()) in scopes[stage]["approved_prompt_sha256"]
    assert execute(live_case, policy=policy)["new_live_requests"] == 0
    assert len(clients) == 5 and not queue


def test_shared_ten_request_limit_covers_multiple_publications(live_case):
    root, _, responses, queue, clients, _ = live_case
    for name in ("live-one", "live-two"):
        queue.extend(responses)
        assert execute(live_case, run_id=name)["new_live_requests"] == 5
    blocked = execute(live_case, run_id="live-three")
    assert blocked["status"] == "extraction_failed" and blocked["new_live_requests"] == 0
    assert len(clients) == 10 and not queue
    with sqlite3.connect(root/"runs/agent/_control/chatai-requests.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 10


def test_failed_attempt_is_counted_and_not_retried(live_case):
    root, _, _, queue, clients, _ = live_case
    queue.append(TimeoutError("private upstream detail"))
    result = execute(live_case)
    assert result["status"] == "extraction_failed" and result["new_live_requests"] == 1
    assert execute(live_case)["new_live_requests"] == 0
    assert len(clients) == 1
    with sqlite3.connect(root/"runs/agent/_control/chatai-requests.sqlite3") as db:
        assert db.execute("SELECT status FROM attempts").fetchall() == [("timeout",)]
    assert "private upstream detail" not in (root/f"runs/agent/{result['run_id']}/sidecar.json").read_text(encoding="utf-8")


def test_unknown_proposal_usage_stops_before_automatic_checker_dispatch(live_case):
    root, spec, responses, queue, clients, _ = live_case
    queue.extend([responses[0], responses[1], (responses[2], None)])
    result = execute(live_case, unknown_usage="stop")
    assert result["status"] == "pending_review" and result["total_model_calls"] == 3
    assert result["usage_unknown"] and result["new_live_requests"] == 3
    assert len(clients) == 3 and not queue
    sidecar = json.loads((root/result["sidecar_relative"]).read_bytes())
    assert sidecar["automatic_patch_status"] == "blocked_unknown_usage"
    assert json.loads(sidecar["workflow_state"]["record"]["canonical_json"]) == json.loads(responses[0])["output"]
    assert not (root/f"runs/agent/{_child(spec,'wc1')}").exists()


def test_unknown_recent_history_blocks_before_sdk_or_credentials(live_case, monkeypatch):
    _, _, _, _, clients, clock = live_case
    # The just-created test ledger has no known history before clock=1000.
    clock[0] = 1000.0
    monkeypatch.delenv("PUBLICATION_TEST_KEY")
    result = execute(live_case)
    assert result["status"] == "extraction_failed" and result["new_live_requests"] == 0
    assert not clients


def test_mismatched_trusted_endpoint_blocks_before_any_run(live_case):
    root, spec, _, _, clients, _ = live_case
    with pytest.raises(ValueError, match="trusted configured"):
        run_publication(root, spec, configured_chatai_endpoint="https://other.invalid/v1")
    assert not (root/f"runs/agent/{spec.run_id}").exists() and not clients


@pytest.mark.parametrize("pattern,path,matches", [
    ("/samples/*/sample_mass", "/samples/0/sample_mass", True),
    ("/samples/*/sample_mass", "/datasets/0/sample_mass", False),
    ("/samples/*/sample_mass", "/samples/arbitrary/sample_mass", False),
    ("/samples/*/sample_mass", "/samples/0/nested/sample_mass", False),
    ("/samples/0/sample_mass", "/samples/1/sample_mass", False),
])
def test_registry_pointer_matching_preserves_entity_collection(pattern, path, matches):
    assert _matches_field(pattern, path) is matches
