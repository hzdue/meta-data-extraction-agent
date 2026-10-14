"""Read-only adapter around the frozen inherited evaluator, for offline use only.

The caller authorizes individual gold/prediction paths in an experiment manifest.
This module neither selects a corpus nor writes reports or modifies the scorer.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from meta_data_extraction import alignment, config_validation, evaluation, normalization, utils
from meta_data_extraction.agent.paths import input_file, workspace_root


class ScoringError(ValueError):
    """An explicit offline scoring failure; exception text contains no record data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class EvaluatorChangedError(ScoringError):
    """The evaluator's code, resources, environment, or origin changed."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FileFingerprint(_Frozen):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DependencyVersion(_Frozen):
    name: str
    version: str


class FieldMetric(_Frozen):
    field: str
    metric: str
    parent_entity: str | None = None


class EvaluatorSnapshot(_Frozen):
    version: Literal["m03-evaluator-1"] = "m03-evaluator-1"
    evaluator_id: str
    files: tuple[FileFingerprint, ...]
    dependencies: tuple[DependencyVersion, ...]
    field_metrics: tuple[FieldMetric, ...]
    python_version: str
    metadata_json: str

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    @model_validator(mode="after")
    def _identity(self) -> EvaluatorSnapshot:
        if self.metadata_json != _canonical(json.loads(self.metadata_json)):
            raise ValueError("Evaluator metadata must use canonical JSON")
        payload = self.model_dump(exclude={"evaluator_id"}, mode="json")
        if self.evaluator_id != "evaluator:" + _digest(_canonical(payload).encode()):
            raise ValueError("Evaluator fingerprint mismatch")
        return self


_MODULES = (evaluation, alignment, normalization, config_validation, utils)
_CODE_FILES = tuple(f"src/meta_data_extraction/{module.__name__.rsplit('.', 1)[1]}.py" for module in _MODULES)
_BASE_FILES = (
    *_CODE_FILES,
    "src/meta_data_extraction/offline/scoring.py",
    "configs/evaluation_metrics.yaml",
    "configs/alignment_helper.yaml",
    "pyproject.toml",
    "uv.lock",
)
_DEPENDENCIES = ("PyYAML", "numpy", "scipy", "pydantic")
_IMPORTED_CODE_HASHES = {
    **{name: _digest(Path(module.__file__).read_bytes()) for module, name in zip(_MODULES, _CODE_FILES)},
    "src/meta_data_extraction/offline/scoring.py": _digest(Path(__file__).read_bytes()),
}


def _origins(root: Path) -> None:
    for module, relative in zip(_MODULES, _CODE_FILES):
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise EvaluatorChangedError("evaluator_module_origin_mismatch")
    if Path(__file__).resolve() != (root / "src/meta_data_extraction/offline/scoring.py").resolve():
        raise EvaluatorChangedError("adapter_module_origin_mismatch")
    if alignment._ALIGNMENT_HELPER_PATH.resolve() != (root / "configs/alignment_helper.yaml").resolve():
        raise EvaluatorChangedError("alignment_helper_origin_mismatch")
    if utils.SCHEMAS_DIR.resolve() != (root / "schemas").resolve():
        raise EvaluatorChangedError("schema_origin_mismatch")
    if any(_digest((root / name).read_bytes()) != value for name, value in _IMPORTED_CODE_HASHES.items()):
        raise EvaluatorChangedError("evaluator_code_changed_since_import")


def _fingerprint(root: Path, relative: str) -> FileFingerprint:
    path = input_file(root, relative, allowed_roots=(relative,))
    return FileFingerprint(path=relative, sha256=_digest(path.read_bytes()))


def _dependencies() -> tuple[DependencyVersion, ...]:
    return tuple(DependencyVersion(name=name, version=importlib.metadata.version(name)) for name in _DEPENDENCIES)


def _field_metrics(plan: dict[str, tuple]) -> tuple[FieldMetric, ...]:
    metrics = []
    for path, spec in plan.items():
        metrics.append(FieldMetric(field=path, metric=spec[0]))
        if spec[0] == "entity":
            metrics.extend(FieldMetric(field=f"{path}.attrs.{name}", metric=child[0], parent_entity=path)
                           for name, child in spec[1].attr_scorers.items())
    return tuple(sorted(metrics, key=lambda item: item.field))


def freeze_evaluator(root: Path | str) -> EvaluatorSnapshot:
    """Freeze effective code, plan, helper, schemas, lockfile and installed versions."""
    root = workspace_root(root)
    _origins(root)
    try:
        initial = tuple(_fingerprint(root, name) for name in _BASE_FILES)
        plan = root / "configs/evaluation_metrics.yaml"
        metadata = evaluation.build_evaluator_metadata(plan)
        field_metrics = _field_metrics(evaluation.load_plan(plan))
        if metadata["schema_version_mismatch"]:
            raise EvaluatorChangedError("schema_version_mismatch")
        schema_names = sorted(metadata["schema_versions_declared"])
        if any(Path(name).name != name or not name.endswith(".json") for name in schema_names):
            raise EvaluatorChangedError("invalid_declared_schema_path")
        files = initial + tuple(_fingerprint(root, f"schemas/{name}") for name in schema_names)
        if initial != tuple(_fingerprint(root, name) for name in _BASE_FILES):
            raise EvaluatorChangedError("evaluator_changed_during_freeze")
        # Paths are provenance, not evaluator semantics. The exact resources above
        # remain mandatory and fixed to this checkout when a snapshot is verified.
        metadata["plan_file"] = "configs/evaluation_metrics.yaml"
        dependencies = _dependencies()
        payload = {
            "version": "m03-evaluator-1",
            "files": [item.model_dump() for item in files],
            "dependencies": [item.model_dump() for item in dependencies],
            "field_metrics": [item.model_dump() for item in field_metrics],
            "python_version": platform.python_version(),
            "metadata_json": _canonical(metadata),
        }
        return EvaluatorSnapshot(
            **{**payload, "files": files, "dependencies": dependencies, "field_metrics": field_metrics},
            evaluator_id="evaluator:" + _digest(_canonical(payload).encode()),
        )
    except ScoringError:
        raise
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        raise EvaluatorChangedError("evaluator_resources_unavailable") from exc


def verify_evaluator(snapshot: EvaluatorSnapshot, root: Path | str) -> None:
    """Reject stale snapshots, even when declared version strings are unchanged."""
    snapshot = EvaluatorSnapshot.model_validate_json(snapshot.model_dump_json())
    if freeze_evaluator(root) != snapshot:
        raise EvaluatorChangedError("evaluator_snapshot_changed")


def _no_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ScoringError("duplicate_json_key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ScoringError("non_finite_json_value")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ScoringError("non_finite_json_value")
    return parsed


def _read_record(path: Path, *, prediction: bool) -> bytes:
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() or parent.is_junction() for parent in path.parents):
        raise ScoringError("linked_scoring_input")
    if not path.is_file() or path.stat().st_nlink > 1:
        raise ScoringError("invalid_scoring_input_file")
    try:
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                            parse_constant=_invalid_constant, parse_float=_finite_float)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScoringError("invalid_json_record") from exc
    if prediction and isinstance(record, dict) and "document_id" in record and "output" in record:
        record = record["output"]
    if not isinstance(record, dict) or not record:
        raise ScoringError("empty_or_non_object_record")
    return raw


def evaluate_pair(
    root: Path | str,
    gold: Path,
    pred: Path,
    snapshot: EvaluatorSnapshot,
    identity_gold: Path | None = None,
) -> dict[str, Any]:
    """Score one explicitly supplied pair, leaving inherited report values intact.

    The inherited matcher uses ``gold.stem`` as the publication identity. Callers
    preserving inputs in a run must preserve that basename, including punctuation.
    """
    root = workspace_root(root)
    verify_evaluator(snapshot, root)
    inputs = [(Path(gold), False), (Path(pred), True)]
    if identity_gold is not None:
        inputs.append((Path(identity_gold), False))
    before = tuple(_digest(_read_record(path, prediction=is_pred)) for path, is_pred in inputs)
    try:
        report = evaluation.evaluate(
            gold, pred, plan_path=root / "configs/evaluation_metrics.yaml", identity_gold_path=identity_gold,
        )
        evaluation.validate_evaluator_provenance(report["evaluator"], snapshot.metadata, source="offline report")
    except Exception as exc:
        raise ScoringError("inherited_evaluation_failed") from exc
    finally:
        verify_evaluator(snapshot, root)
    after = tuple(_digest(_read_record(path, prediction=is_pred)) for path, is_pred in inputs)
    if before != after:
        raise ScoringError("scoring_inputs_changed")
    return report


class ParityResult(_Frozen):
    status: Literal["matched", "mismatch"]
    differences: tuple[str, ...] = ()


def _semantic_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or any(
        not isinstance(report.get(name), dict) for name in ("fields", "evaluator", "identity_sources")
    ):
        raise ScoringError("invalid_report_shape")
    # Copy via JSON to avoid mutating either released reports or new results.
    result = json.loads(_canonical(report))
    for name in ("gold", "pred", "plan"):
        result.pop(name, None)
    if isinstance(result.get("evaluator"), dict):
        result["evaluator"].pop("plan_file", None)
    if isinstance(result.get("identity_sources"), dict):
        for source in result["identity_sources"].values():
            if isinstance(source, dict):
                source.pop("path", None)
    return result


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(left.keys() | right.keys()):
            pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in left or key not in right:
                differences.append(pointer)
            else:
                differences.extend(_differences(left[key], right[key], pointer))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        return [part for i, (a, b) in enumerate(zip(left, right)) for part in _differences(a, b, f"{path}/{i}")]
    return [] if type(left) is type(right) and left == right else [path or "/"]


def parity(report: dict[str, Any], reference: dict[str, Any]) -> ParityResult:
    """Compare every semantic report value exactly, ignoring only known paths."""
    differences = tuple(_differences(_semantic_report(report), _semantic_report(reference)))
    return ParityResult(status="mismatch" if differences else "matched", differences=differences)


class MetricRow(_Frozen):
    field: str
    category: Literal["entity_recovery", "conditional_attributes", "references", "scalar_fields"]
    metrics: dict[str, Any]
    support_status: Literal["known", "zero", "unknown"]
    support_count: int | None
    support_unit: Literal["gold_entities", "gold_values", "matched_gold_values", "gold_references", "gold_set_items"]
    n_papers: int
    n_pairs: int | None = None


def summaries(reports: list[dict[str, Any]], snapshot: EvaluatorSnapshot | None = None) -> tuple[MetricRow, ...]:
    """Keep inherited aggregates and expose their actual conditional support.

    Means include gold-null agreements exactly as inherited. ``support_count``
    exposes non-null gold values; it is not a new non-null-only score. References
    remain conditional on matched parent entities. No cross-field mean is added.
    """
    if snapshot is None:
        snapshot = freeze_evaluator(Path(__file__).resolve().parents[3])
    field_specs = {item.field: item for item in snapshot.field_metrics}
    aggregates = evaluation.aggregate_reports(reports)
    rows = []
    for field, metrics in sorted(aggregates.items()):
        values = [report["fields"][field] for report in reports if field in report["fields"]]
        if field not in field_specs:
            raise ScoringError("report_field_not_in_frozen_plan")
        spec = field_specs[field]
        attribute = spec.parent_entity is not None
        reference = spec.metric == "reference"
        entity = spec.metric == "entity"
        count_metric = spec.metric in {"entity", "reference", "set_f1", "ordered_set_f1"}
        pairs = sum(value["n_pairs"] for value in values) if attribute and all("n_pairs" in value for value in values) else None
        if entity:
            category, unit, key = "entity_recovery", "gold_entities", "n_gold"
        elif reference:
            category, unit, key = "references", "gold_references", None
        elif attribute:
            category = "conditional_attributes"
            unit, key = ("gold_set_items", None) if count_metric else ("matched_gold_values", "n_pairs_non_null")
        else:
            category = "scalar_fields"
            unit, key = ("gold_set_items", None) if count_metric else ("gold_values", "gold_null")
        if attribute and pairs == 0:
            support = 0
        elif key == "gold_null":
            support = sum(not value.get("gold_null", False) for value in values) if all("score" in value for value in values) else None
        elif key is not None:
            support = sum(value.get(key, 0) for value in values) if all(key in value or value == {"n_pairs": 0} for value in values) else None
        else:
            support = sum(value.get("tp", 0) + value.get("fn", 0) for value in values) if all("tp" in value and "fn" in value or value == {"n_pairs": 0} for value in values) else None
        rows.append(MetricRow(
            field=field, category=category, metrics=metrics,
            support_status="unknown" if support is None else ("zero" if support == 0 else "known"),
            support_count=support, support_unit=unit, n_papers=len(values), n_pairs=pairs,
        ))
    return tuple(rows)
