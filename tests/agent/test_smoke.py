import ast
import hashlib
from importlib.util import resolve_name
import json
from pathlib import Path
import socket

import pytest
import meta_data_extraction

from meta_data_extraction.agent.smoke import main, run_smoke


def test_imported_package_belongs_to_this_checkout():
    repository = Path(__file__).resolve().parents[2]
    assert Path(meta_data_extraction.__file__).resolve().parent == (
        repository / "src/meta_data_extraction"
    )


def test_synthetic_smoke_is_offline_and_preserves_releases(workspace):
    release = workspace / "outputs/paper/results.json"
    release.parent.mkdir(parents=True)
    release.write_text("synthetic released sentinel")
    result = run_smoke(workspace, "configs/agent/offline.yaml")
    assert result == workspace / "runs/agent/m00-smoke/smoke.json"
    payload = json.loads(result.read_text())
    assert payload["kind"] == "m00_input_smoke"
    assert payload["model_calls"] == 0
    assert payload["extraction_performed"] is False
    assert "output" not in payload
    assert len(payload["sources"]) == 1
    source = workspace / payload["sources"][0]["path"]
    assert payload["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["schema_top_level_keys"] == [
        "bibliographic", "datasets", "document_overview", "facilities", "samples",
    ]
    assert release.read_text() == "synthetic released sentinel"
    with pytest.raises(FileExistsError):
        run_smoke(workspace, "configs/agent/offline.yaml")


def test_cli_reports_reused_run_as_failure(workspace, monkeypatch, capsys):
    monkeypatch.chdir(workspace)
    assert main(["--run-id", "cli-check"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--run-id", "cli-check"])
    assert exc.value.code == 2
    assert "M00 smoke failed" in capsys.readouterr().err


def test_network_guard_is_active():
    with pytest.raises(AssertionError, match="Network access"):
        socket.create_connection(("invalid.example", 443))


def test_runtime_import_closure_excludes_offline_evaluation_and_limits_live_clients():
    """Only M02 providers may reach the inherited live adapter and named SDK types."""
    src = Path(__file__).resolve().parents[2] / "src"
    pending = []
    for path in (src / "meta_data_extraction/agent").rglob("*.py"):
        parts = path.relative_to(src).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        pending.append(".".join(parts))
    visited = set()
    forbidden = {
        "meta_data_extraction.alignment", "meta_data_extraction.evaluation",
        "meta_data_extraction.normalization", "meta_data_extraction.pipeline",
        "meta_data_extraction.offline",
        "meta_data_extraction.components.llm", "scripts", "openai", "arxiv",
    }
    allowed_live_imports = {
        "meta_data_extraction.agent.providers": {
            "meta_data_extraction.components.llm",
            "meta_data_extraction.components.llm.OpenAICompatibleLLM",
            "openai", "openai.DefaultHttpxClient", "openai.APITimeoutError",
            "openai.RateLimitError", "openai.APIStatusError",
        },
        "meta_data_extraction.components.llm": {"openai", "openai.OpenAI"},
    }
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = src / (module.replace(".", "/") + ".py")
        if not path.is_file():
            path = src / module.replace(".", "/") / "__init__.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
                imported_module = resolve_name("." * node.level + (node.module or ""), package)
                imports = [imported_module]
                imports.extend(f"{imported_module}.{alias.name}" for alias in node.names)
            for imported in imports:
                restricted = any(imported == bad or imported.startswith(bad + ".") for bad in forbidden)
                assert not restricted or imported in allowed_live_imports.get(module, set()), (
                    f"Runtime import {module} -> {imported} crosses the offline/provider boundary"
                )
                if imported.startswith("meta_data_extraction") and (
                    (src / (imported.replace(".", "/") + ".py")).is_file()
                    or (src / imported.replace(".", "/") / "__init__.py").is_file()
                ):
                    pending.append(imported)
    assert "meta_data_extraction.utils" in visited
    assert "meta_data_extraction.config_validation" in visited
    assert "meta_data_extraction.components.llm" in visited
