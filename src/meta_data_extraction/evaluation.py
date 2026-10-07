"""Evaluate extracted metadata against gold annotations.

Use evaluate for one JSON pair and aggregate_reports for a corpus.
Entity matching lives in alignment.py; scoring uses canonicalized source values
with configured normalizers and remaps references through alignments.
"""


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from meta_data_extraction import alignment
from meta_data_extraction.config_validation import check_keys, load_yaml_mapping
from meta_data_extraction.normalization import NORMALIZERS, canonicalize_strings, tokenize
from meta_data_extraction.utils import get_schema_refs


DEFAULT_PLAN_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "evaluation_metrics.yaml"
)


# Main workflow: load records, align all entities, then score.


def evaluate(
    gold_path: str | Path,
    pred_path: str | Path,
    plan_path: str | Path | None = None,
    identity_gold_path: str | Path | None = None,
) -> dict:
    """Load two JSON records, align their entities, score, and assemble a report.

    Supply reconciled ``identity_gold_path`` when comparing two annotators.
    """
    effective_plan = plan_path or DEFAULT_PLAN_PATH
    plan_data, plan = _load_plan_file(effective_plan)
    evaluator_block = build_evaluator_metadata(effective_plan, plan_data=plan_data)

    gold_path = Path(gold_path)
    pred_path = Path(pred_path)
    identity_path = Path(identity_gold_path or gold_path)

    gold = canonicalize_strings(json.loads(gold_path.read_text(encoding="utf-8")))
    if identity_path.resolve() == gold_path.resolve():
        identity_gold = gold
    else:
        identity_gold = canonicalize_strings(
            json.loads(identity_path.read_text(encoding="utf-8"))
        )
    pred_raw = json.loads(pred_path.read_text(encoding="utf-8"))
    pred = canonicalize_strings(_unwrap_extraction_result(pred_raw))

    alignments = _align_document(pred, gold, plan, identity_gold, gold_path.stem)
    fields = _score_document(pred, gold, plan, alignments)

    return {
        "gold": str(gold_path),
        "pred": str(pred_path),
        "plan": str(effective_plan),
        "evaluator": evaluator_block,
        "identity_sources": {
            field: {"path": str(identity_path)}
            for field in ("bibliographic.funding", "samples", "datasets")
        },
        "fields": fields,
    }


# Input and plan preparation.


def _unwrap_extraction_result(pred: Any) -> Any:
    """Accept either an ExtractionResult wrapper or a plain annotation record."""
    if isinstance(pred, dict) and "output" in pred and "document_id" in pred:
        return pred["output"]
    return pred


# Each compiled field uses one scoring tuple:
# (metric,) or (metric, normalizer) for scalar and set metrics;
# ("reference", target_path) for IDs referring to an entity collection;
# ("entity", EntityConfig) for entity recovery and aligned attributes.


@dataclass(frozen=True)
class EntityConfig:
    key_fn: Callable[[dict, dict], float] | None
    attr_scorers: dict[str, tuple]
    tau: float = 0.5
    matcher: str = "similarity"


def load_plan(plan_path: str | Path) -> dict[str, tuple]:
    """Read and validate the YAML plan, returning field paths and scoring tuples."""
    _, plan = _load_plan_file(plan_path)
    return plan


def _load_plan_file(plan_path: str | Path) -> tuple[dict, dict[str, tuple]]:
    """Read and validate the plan once, returning its metadata and dispatch."""
    source = str(plan_path)
    data = load_yaml_mapping(Path(plan_path).read_text(encoding="utf-8"), source=source)
    check_keys(data, {"version", "schema_versions", "scoring_plan"}, source)
    if not isinstance(data.get("version"), str) or not data["version"].strip():
        raise ValueError(f"{source}: version must be a non-empty version string")
    schemas = data.get("schema_versions", {})
    if not isinstance(schemas, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in schemas.items()
    ):
        raise ValueError(f"{source}: schema_versions must map schema names to version strings")
    return data, _compile_plan(data)


def _compile_plan(data: dict) -> dict[str, tuple]:
    """Compile field metrics and check that references name entity collections."""
    plan = data.get("scoring_plan")
    if not isinstance(plan, dict) or not plan:
        raise ValueError("scoring_plan must be a non-empty mapping")
    compiled = {}
    for path, node in plan.items():
        if not isinstance(path, str) or not path:
            raise ValueError("scoring_plan field paths must be non-empty strings")
        compiled[path] = _spec_from_yaml(node)
    for path, spec in compiled.items():
        fields = spec[1].attr_scorers.values() if spec[0] == "entity" else [spec]
        for child in fields:
            if child[0] == "reference":
                target = child[1]
                if target not in compiled or compiled[target][0] != "entity":
                    raise ValueError(f"{path}: reference target {target!r} must name an entity field")
    return compiled


_ENTITY_MATCHERS = (
    "similarity", "facility_staged", "funding_registry", "sample_registry", "dataset_registry",
)


def _spec_from_yaml(node: dict) -> tuple:
    """Validate a YAML metric and return its scoring tuple."""
    if not isinstance(node, dict) or not isinstance(node.get("metric"), str):
        raise ValueError("metric specification must be a mapping with a metric name")
    metric = node["metric"]
    if metric == "exact":
        check_keys(node, {"metric"}, "exact metric")
        return (metric,)
    if metric in _NORMALIZED_METRICS:
        check_keys(node, {"metric", "normalizer"}, f"{metric} metric")
        norm = _resolve_normalizer(node.get("normalizer"))
        return (metric,) if norm is None else (metric, norm)
    if metric == "reference":
        check_keys(node, {"metric", "target"}, "reference metric")
        if not isinstance(node.get("target"), str) or not node["target"]:
            raise ValueError("reference target must be a non-empty field path")
        return (metric, node["target"])
    if metric == "entity":
        matcher = node.get("matcher", "similarity")
        if matcher not in _ENTITY_MATCHERS:
            raise ValueError(f"unknown entity matcher: {matcher!r}")
        allowed = {"metric", "matcher", "attrs"}
        if matcher == "similarity":
            allowed.update({"key_attrs", "tau", "key_normalizer"})
        elif matcher == "facility_staged":
            allowed.add("tau")
        check_keys(node, allowed, "entity metric")
        if not isinstance(node.get("attrs"), dict):
            raise ValueError("entity attrs must be a mapping")
        if "tau" in node:
            tau = node["tau"]
            if isinstance(tau, bool) or not isinstance(tau, (float, int)) or not 0 <= tau <= 1:
                raise ValueError("entity tau must be a number between 0 and 1")
        key_fn = None
        if matcher == "similarity":
            keys = node.get("key_attrs")
            if (
                not isinstance(keys, list)
                or not keys
                or any(not isinstance(key, str) or not key for key in keys)
            ):
                raise ValueError("entity key_attrs must be a non-empty list of attribute names")
            key_fn = alignment.make_key_scorer(
                *keys,
                normalizer=_resolve_normalizer(node.get("key_normalizer")),
            )
        attrs = {attr: _spec_from_yaml(child) for attr, child in node["attrs"].items()}
        if any(spec[0] == "entity" for spec in attrs.values()):
            raise ValueError("entity attributes must use scalar, set, or reference metrics")
        config = EntityConfig(
            key_fn=key_fn,
            attr_scorers=attrs,
            tau=node.get("tau", 0.5),
            matcher=matcher,
        )
        return ("entity", config)
    raise ValueError(f"unknown metric kind in YAML: {metric!r}")


def _resolve_normalizer(name: str | None) -> Callable | None:
    if name is None:
        return None
    if name not in NORMALIZERS:
        raise ValueError(f"unknown normalizer: {name!r}. Known: {sorted(NORMALIZERS)}")
    return NORMALIZERS[name]


# Document scoring: alignments, entity recovery, attributes, and references.


def _align_document(
    pred: dict,
    gold: dict,
    plan: dict[str, tuple],
    identity_gold: dict,
    document_id: str,
) -> dict[str, alignment.AlignmentResult]:
    """Call each entity matcher before scoring; IAA uses reconciled identities."""
    alignments = {}
    for path, spec in plan.items():
        if spec[0] != "entity":
            continue
        config = spec[1]
        p, g = _field_value(pred, path), _field_value(gold, path)
        if config.matcher == "similarity":
            result = alignment.align_similarity(p, g, key_fn=config.key_fn, tau=config.tau)
        elif config.matcher == "facility_staged":
            result = alignment.align_facilities(p, g, tau=config.tau)
        elif config.matcher == "funding_registry":
            result = alignment.align_funding(
                p, g, identity_gold=_field_value(identity_gold, "bibliographic.funding") or []
            )
        elif config.matcher == "sample_registry":
            result = alignment.align_samples(
                p, g, document_id=document_id,
                identity_gold=identity_gold.get("samples") or [],
            )
        else:  # dataset_registry; matcher names are validated when loading the plan.
            result = alignment.align_datasets(
                p, g, document_id=document_id,
                identity_gold=identity_gold.get("datasets") or [],
            )
        alignments[path] = result
    return alignments


def _score_document(
    pred: dict,
    gold: dict,
    plan: dict[str, tuple],
    alignments: dict[str, alignment.AlignmentResult],
) -> dict[str, dict]:
    """Score direct fields, entity recovery, aligned attributes, and references."""
    fields = {}
    for path, spec in plan.items():
        pred_value, gold_value = _field_value(pred, path), _field_value(gold, path)
        if spec[0] == "entity":
            result = _score_entities(
                pred_value, gold_value, spec[1], alignments[path], alignments
            )
            attrs = result.pop("attrs")
            fields[path] = result
            for attr, scores in attrs.items():
                fields[f"{path}.attrs.{attr}"] = scores
        else:
            fields[path] = _score_value(spec, pred_value, gold_value, alignments)
    return fields


def _score_entities(
    pred: list[dict] | None,
    gold: list[dict] | None,
    config: EntityConfig,
    aligned: alignment.AlignmentResult,
    alignments: dict[str, alignment.AlignmentResult],
) -> dict:
    """Score entity recovery and original attribute values on matched pairs."""
    pred = pred or []
    gold = gold or []
    matches = aligned.pairs
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gold) - tp
    attrs = {}
    for path, spec in config.attr_scorers.items():
        pair_scores = []
        for match in matches:
            pred_value = _field_value(pred[match.pred_index], path)
            gold_value = _field_value(gold[match.gold_index], path)
            pair_scores.append(_score_value(spec, pred_value, gold_value, alignments))
        attrs[path] = _aggregate_attribute_scores(pair_scores)
    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "f1": _f1_from_counts(tp, fp, fn),
        "n_pred": len(pred), "n_gold": len(gold),
        "attrs": attrs,
        **aligned.diagnostics,
    }
    return result


def _aggregate_attribute_scores(results: list[dict]) -> dict:
    """Pool count metrics or average scalar scores across matched entity pairs."""
    if not results:
        return {"n_pairs": 0}
    first = results[0]
    if "tp" in first:
        tp = sum(r.get("tp", 0) for r in results)
        fp = sum(r.get("fp", 0) for r in results)
        fn = sum(r.get("fn", 0) for r in results)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": _f1_from_counts(tp, fp, fn),
            "n_pairs": len(results),
        }
    scores = [r["score"] for r in results]
    return {
        "mean_score": sum(scores) / len(scores),
        "n_pairs": len(scores),
        "n_pairs_non_null": sum(not r.get("gold_null") for r in results),
    }


def _score_value(
    spec: tuple, pred: Any, gold: Any, alignments: dict[str, alignment.AlignmentResult]
) -> dict:
    """Score a scalar, set, or reference field using completed alignments."""
    kind = spec[0]
    if kind == "exact":
        return _exact(pred, gold)
    if kind in _NORMALIZED_METRICS:
        normalizer = spec[1] if len(spec) > 1 else None
        return _NORMALIZED_METRICS[kind](pred, gold, normalizer=normalizer)
    if kind == "reference":
        return _score_references(pred, gold, alignments[spec[1]].id_map)
    raise ValueError(f"unknown value metric: {kind!r}")


_UNMATCHED_REFERENCE = object()


def _score_references(
    pred_refs: list[str] | None,
    gold_refs: list[str] | None,
    alignment: dict[str, str | None],
) -> dict:
    """Remap predicted IDs to gold IDs and score reference sets, including misses."""
    pred_refs = pred_refs or []
    gold_refs = gold_refs or []
    remapped: list[Any] = []
    for ref in pred_refs:
        mapped = alignment.get(ref)
        if mapped is not None:
            remapped.append(mapped)
        else:
            # A tuple cannot collide with a schema-valid gold reference ID.
            # Including the source ID preserves set semantics: a repeated
            # dangling reference counts once, while two distinct references
            # count as two false positives.
            remapped.append((_UNMATCHED_REFERENCE, ref))
    return _set_f1(remapped, gold_refs)


def _field_value(doc: Any, path: str) -> Any:
    """Read a dotted field path; missing fields have value None."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# Metric calculations: equality, text overlap, set recovery, and heading order.


def _exact(pred: Any, gold: Any) -> dict:
    edge = _score_missing_values(pred, gold)
    if edge is not None:
        return edge
    return {"score": 1.0 if pred == gold else 0.0}


def _normalized_exact(
    pred: Any, gold: Any, *, normalizer: Callable[[Any], Any] | None = None
) -> dict:
    if normalizer is not None and isinstance(pred, str) and isinstance(gold, str):
        pred = normalizer(pred)
        gold = normalizer(gold)
    return _exact(pred, gold)


def _rouge_l(
    pred: Any,
    gold: Any,
    *,
    normalizer: Callable[[Any], Any] | None = None,
) -> dict:
    edge = _score_missing_values(pred, gold)
    if edge is not None:
        return edge
    if normalizer is not None and isinstance(pred, str) and isinstance(gold, str):
        pred = normalizer(pred)
        gold = normalizer(gold)
    p_tok = tokenize(str(pred))
    g_tok = tokenize(str(gold))
    if not p_tok and not g_tok:
        return {"score": 1.0}
    if not p_tok or not g_tok:
        return {"score": 0.0}
    lcs = _lcs_length(p_tok, g_tok)
    if lcs == 0:
        return {"score": 0.0}
    p = lcs / len(p_tok)
    r = lcs / len(g_tok)
    return {"score": 2 * p * r / (p + r)}


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _score_missing_values(pred: Any, gold: Any) -> dict | None:
    """Both null scores 1; one null scores 0. Record gold support separately."""
    if pred is None and gold is None:
        return {"score": 1.0, "gold_null": True}
    if gold is None:
        return {"score": 0.0, "gold_null": True}
    if pred is None:
        return {"score": 0.0}
    return None


def _set_f1(
    pred: list | None,
    gold: list | None,
    *,
    normalizer: Callable[[Any], Any] | None = None,
) -> dict:
    pred_list = pred if isinstance(pred, list) else ([] if pred is None else [pred])
    gold_list = gold if isinstance(gold, list) else ([] if gold is None else [gold])
    norm = normalizer or (lambda x: x)
    P = {norm(x) if isinstance(x, str) else x for x in pred_list}
    G = {norm(x) if isinstance(x, str) else x for x in gold_list}
    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": _f1_from_counts(tp, fp, fn)}


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def _ordered_set_f1(
    pred: list | None,
    gold: list | None,
    *,
    normalizer: Callable[[Any], Any] | None = None,
) -> dict:
    content = _set_f1(pred, gold, normalizer=normalizer)
    pred = pred or []
    gold = gold or []
    norm = normalizer or (lambda x: x)
    p_normed = [norm(x) for x in pred]
    g_normed = [norm(x) for x in gold]
    g_set = set(g_normed)
    p_set = set(p_normed)
    pred_match = [x for x in p_normed if x in g_set]
    gold_match = [x for x in g_normed if x in p_set]
    if len(pred_match) < 2 or len(gold_match) < 2:
        return {**content, "order": None}
    gold_pos = {x: i for i, x in enumerate(gold_match)}
    seq = [gold_pos[x] for x in pred_match]
    n = len(seq)
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
    tau = 1 - 2 * inv / (n * (n - 1) / 2)
    return {**content, "order": tau}


_NORMALIZED_METRICS = {
    "normalized_exact": _normalized_exact,
    "rouge_l": _rouge_l,
    "set_f1": _set_f1,
    "ordered_set_f1": _ordered_set_f1,
}


# Corpus aggregation: pool counts and average scores across papers.


def aggregate_reports(reports: list[dict]) -> dict[str, dict]:
    """Pool TP/FP/FN counts, average scalar scores, and pair-weight attributes."""
    by_field: dict[str, list[dict]] = {}
    for r in reports:
        for path, result in r["fields"].items():
            by_field.setdefault(path, []).append(result)

    aggregated: dict[str, dict] = {}
    for path, results in by_field.items():
        # Empty-pair placeholders do not identify the metric's result shape.
        first = next((r for r in results if set(r.keys()) != {"n_pairs"}), results[0])
        if "tp" in first:
            tp = sum(r.get("tp", 0) for r in results)
            fp = sum(r.get("fp", 0) for r in results)
            fn = sum(r.get("fn", 0) for r in results)
            aggregated[path] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": _f1_from_counts(tp, fp, fn),
                "n_papers": len(results),
            }
            abstention_keys = sorted(
                {key for r in results for key in r.get("alignment_abstentions", {})}
            )
            if abstention_keys:
                aggregated[path]["alignment_abstentions"] = {
                    key: sum(
                        r.get("alignment_abstentions", {}).get(key, 0) for r in results
                    )
                    for key in abstention_keys
                }
            # Ordering is defined only for papers with at least two matches.
            if any("order" in r for r in results):
                orders = [r["order"] for r in results if r.get("order") is not None]
                aggregated[path]["mean_order"] = (
                    sum(orders) / len(orders) if orders else None
                )
                aggregated[path]["n_ordered"] = len(orders)
        elif "score" in first:
            scores = [r["score"] for r in results]
            aggregated[path] = {
                "mean_score": sum(scores) / len(scores),
                "n_papers": len(scores),
                "n_non_null": sum(not r.get("gold_null") for r in results),
            }
        elif "mean_score" in first:
            total_pairs = sum(r.get("n_pairs", 0) for r in results)
            weighted = sum(
                r.get("mean_score", 0.0) * r.get("n_pairs", 0) for r in results
            )
            aggregated[path] = {
                "mean_score": weighted / total_pairs,
                "n_pairs": total_pairs,
                "n_papers": len(results),
                "n_pairs_non_null": sum(r.get("n_pairs_non_null", 0) for r in results),
            }
        else:
            aggregated[path] = {"n_papers": len(results)}
    return aggregated


# Report metadata: declare and validate the evaluator's configuration versions.


def build_evaluator_metadata(
    plan_path: str | Path, *, plan_data: dict | None = None
) -> dict[str, Any]:
    """Build the ``evaluator`` header block stamped into each report."""
    if plan_data is None:
        plan_data, _ = _load_plan_file(plan_path)
    declared = dict(plan_data.get("schema_versions", {}) or {})
    actual = get_schema_refs(list(declared.keys())) if declared else {}
    return {
        "plan_file": str(plan_path),
        "plan_version": str(plan_data["version"]),
        "alignment_helper_version": alignment.load_alignment_helper()["version"],
        "schema_versions_declared": declared,
        "schema_versions_actual": actual,
        "schema_version_mismatch": declared != actual,
    }


def validate_evaluator_provenance(evaluator: dict, expected: dict, *, source: str) -> None:
    """Check the declared plan, alignment-helper, and schema versions."""
    labels = {
        "plan_version": "plan version",
        "alignment_helper_version": "alignment helper version",
        "schema_versions_declared": "declared schemas",
        "schema_versions_actual": "actual schemas",
    }
    for key, label in labels.items():
        if key not in evaluator or evaluator[key] != expected[key]:
            raise ValueError(
                f"{source}: unexpected evaluator provenance: missing or stale {label} "
                f"{evaluator.get(key)!r}; expected {expected[key]!r}"
            )
    if evaluator.get("schema_version_mismatch") is not False:
        raise ValueError(f"{source}: unexpected evaluator provenance: schema version mismatch")
