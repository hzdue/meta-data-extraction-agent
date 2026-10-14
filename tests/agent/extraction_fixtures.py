"""Invented M02 facts and deterministic responses, never measured extraction."""
from meta_data_extraction.agent.contracts import assertion_values
from meta_data_extraction.agent.evidence import ParserProvenance, bundle_document
from meta_data_extraction.data_models import Document
from meta_data_extraction.utils import build_skeleton


SYNTHETIC_TEXT = (
    "Synthetic diffraction study. A. Example works at Synthetic Institute and is not a "
    "corresponding author. Sample A and Sample B are distinct specimens, both with the "
    "stated composition CaCO3. Dataset A measures both samples using X-ray diffraction "
    "at Synthetic Beam Source. This synthetic document explicitly has no references, "
    "figures, or tables.\n"
)


def synthetic_record(profile):
    record = build_skeleton(profile.schema)
    record["bibliographic"]["title"] = "Synthetic diffraction study"
    record["bibliographic"]["affiliations"] = [
        {"id": "aff_1", "label": None, "text": "Synthetic Institute"},
    ]
    record["bibliographic"]["authors"] = [{
        "id": "author_1", "full_name": "A. Example", "orcid": None,
        "email": None, "is_corresponding": False, "affiliation_refs": ["aff_1"],
    }]
    record["facilities"] = [{"id": "facility_1", "name": "Synthetic Beam Source"}]
    for index, name in enumerate(("Sample A", "Sample B"), 1):
        sample = build_skeleton(profile.schema["samples"]["item_schema"])
        sample.update(id=f"sample_{index}", sample_name=name, chemical_composition="CaCO3")
        record["samples"].append(sample)
    dataset = build_skeleton(profile.schema["datasets"]["item_schema"])
    dataset.update(id="dataset_1", dataset_name="Dataset A", sample_refs=["sample_1", "sample_2"],
                   facility_refs=["facility_1"], measurement_technique="X-ray diffraction")
    record["datasets"] = [dataset]
    for field in ("has_references", "has_figures", "has_tables"):
        record["document_overview"][field] = False
    return record


def synthetic_bundle(text=SYNTHETIC_TEXT, *, source_id="synthetic-m02.md", max_chars=None):
    return bundle_document(
        Document(id="synthetic", text=text), source_id=source_id,
        source_bytes=text.encode("utf-8"), source_format="markdown",
        parser=ParserProvenance(name="synthetic", version="1"), max_chars=max_chars,
    )


def evidence_response(record, bundle):
    """Link each synthetic assertion to its paragraph; support stays not_assessed."""
    region = bundle.regions[0]
    return {"output": record, "evidence": [
        {"path": path, "region_id": region.region_id, "start": region.start,
         "end": region.end, "quotation": bundle.text[region.start:region.end]}
        for path, _ in assertion_values(record)
    ]}
