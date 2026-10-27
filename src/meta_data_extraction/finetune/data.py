"""Deterministic D/E/R projections and prompts, with no model dependencies."""
from __future__ import annotations

from copy import deepcopy
import json

from meta_data_extraction.utils import build_prompt, load_merged_schema

COLLECTIONS = ("facilities", "samples", "datasets")
REFERENCE_FIELDS = ("sample_refs", "facility_refs")


def canonical_json(value) -> str:
    """Stable whitespace, Unicode, original object/array ordering, no NaN."""
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def project_entities(gold: dict) -> dict:
    result = {key: deepcopy(gold[key]) for key in COLLECTIONS}
    for dataset in result["datasets"]:
        for field in REFERENCE_FIELDS:
            del dataset[field]
    return result


def project_references(gold: dict) -> dict:
    return {"references": [
        {"dataset_id": dataset["id"], "field": field, "target_ids": list(dataset[field])}
        for dataset in gold["datasets"] for field in REFERENCE_FIELDS
    ]}


def reconstruct(entities: dict, references: dict) -> dict:
    result = deepcopy(entities)
    datasets = {item["id"]: item for item in result["datasets"]}
    if len(datasets) != len(result["datasets"]):
        raise ValueError("Duplicate dataset IDs")
    endpoints = {"sample_refs": {x["id"] for x in entities["samples"]},
                 "facility_refs": {x["id"] for x in entities["facilities"]}}
    expected = [(dataset["id"], field) for dataset in result["datasets"] for field in REFERENCE_FIELDS]
    actual = [(row["dataset_id"], row["field"]) for row in references["references"]]
    if actual != expected:
        raise ValueError("Reference records must cover each dataset/field once, in order")
    for row in references["references"]:
        refs = row["target_ids"]
        if not isinstance(refs, list) or len(refs) != len(set(refs)) or not set(refs) <= endpoints[row["field"]]:
            raise ValueError("Duplicate, invalid, or dangling reference target")
        datasets[row["dataset_id"]][row["field"]] = list(refs)
    return result


def task_prompt(task: str, text: str, schemas: list[str], entities: dict | None = None) -> str:
    if task == "D":
        return build_prompt(schemas, text)
    schema = load_merged_schema(schemas)
    if task == "E":
        task_text = (
            "Extract the facilities, samples, and datasets from the document. Return exactly "
            "these three top-level lists. Include each item's id and all its non-reference "
            "attributes from the schema. Omit sample_refs and facility_refs entirely from "
            "datasets. Do not return bibliographic or document_overview fields. Assign IDs "
            "using the schema conventions."
        )
        output_text = '{"facilities": [], "samples": [], "datasets": []}'
        inventory = ""
    elif task == "R":
        if entities is None:
            raise ValueError("R requires the reference-free entity inventory")
        if any(field in dataset for dataset in entities["datasets"] for field in REFERENCE_FIELDS):
            raise ValueError("Reference answers must not appear in the R inventory")
        task_text = (
            "Resolve dataset-to-sample and dataset-to-facility links using the document "
            "and the supplied entity inventory. Do not extract new entities or change their "
            "IDs. Return a references list containing exactly two records per dataset, in "
            "inventory dataset order: sample_refs first, then facility_refs. Each record "
            "must contain dataset_id, field, and target_ids. Use only supplied IDs; preserve "
            "target order as specified by the schema. Include target_ids: [] when no link "
            "is supported."
        )
        output_text = '{"references": [{"dataset_id": "dataset_1", "field": "sample_refs", "target_ids": []}, {"dataset_id": "dataset_1", "field": "facility_refs", "target_ids": []}]}'
        inventory = "## Entity inventory (reference fields removed)\n\n" + canonical_json(entities) + "\n\n"
    else:
        raise ValueError(f"Unknown task: {task}")
    return (
        "## Task\nYou are an expert system for extracting metadata from scientific documents.\n\n"
        + task_text + "\n\n## Full source schema\n\n" + json.dumps(schema, indent=2)
        + "\n\n## Rules\n\n"
        "- Use only information explicitly stated in the document; do not guess.\n"
        "- Preserve values, nulls, empty lists and ordering under the schema rules.\n"
        "- The task above selects a projection of the full source schema; return only that projection.\n"
        "- Return a single valid JSON object, without code fences, explanations or reasoning.\n\n"
        "## Output shape\n\n" + output_text + "\n\n" + inventory + "## Document\n\n" + text
    )


def make_record(paper_id: str, split: str, task: str, text: str, gold: dict,
                schemas: list[str], provenance: dict) -> dict:
    if split not in {"train", "validation"} or (split == "validation" and task != "D"):
        raise ValueError("Only train D/E/R and validation D targets are permitted")
    entities, references = project_entities(gold), project_references(gold)
    if reconstruct(entities, references) != {key: gold[key] for key in COLLECTIONS}:
        raise ValueError("E + R does not reconstruct the gold domain collections")
    target = {"D": gold, "E": entities, "R": references}[task]
    return {
        "example_id": f"{paper_id}:{task}", "paper_id": paper_id, "split": split, "task": task,
        "prompt": [{"role": "user", "content": task_prompt(task, text, schemas, entities)}],
        "completion": [{"role": "assistant", "content": canonical_json(target), "reasoning_content": ""}],
        "provenance": deepcopy(provenance),
    }
