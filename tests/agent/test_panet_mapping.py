"""Synthetic mapping diagnostics only; no real mapping approval or gold labels."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from extraction_fixtures import synthetic_bundle, synthetic_record
from meta_data_extraction.agent.contracts import digest, snapshot
from meta_data_extraction.agent.evidence import evidence_ref, sha256
from meta_data_extraction.agent.panet_lookup import lexical_candidates
from meta_data_extraction.agent.panet_mapping import (
    MappingDecision, TechniqueMention, propose_mapping, verify_mapping,
)
from meta_data_extraction.agent.panet_resources import ResourceSpec, load_panet
from meta_data_extraction.agent.profiles import load_profile

NS = "urn:synthetic:m04-map:"
FAMILY = NS + "root"


def synthetic_ontology(workspace, *, extra_classes=""):
    raw = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#"
 xmlns:obo="http://purl.obolibrary.org/obo/"
 xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#"
 xmlns:skos="http://www.w3.org/2004/02/skos/core#">
 <owl:Ontology rdf:about="{NS}ontology"><owl:versionInfo>synthetic-v1</owl:versionInfo></owl:Ontology>
 <owl:Class rdf:about="{FAMILY}"><rdfs:label>diffraction techniques</rdfs:label></owl:Class>
 <owl:Class rdf:about="{NS}xrd"><rdfs:label>X-ray diffraction</rdfs:label>
  <rdfs:subClassOf rdf:resource="{FAMILY}"/>
  <oboInOwl:hasExactSynonym>XRD</oboInOwl:hasExactSynonym>
  <skos:altLabel>diffraction by X-rays</skos:altLabel>
  <oboInOwl:hasRelatedSynonym>diffraction probe</oboInOwl:hasRelatedSynonym>
  <oboInOwl:hasBroadSynonym>scattering measurement</oboInOwl:hasBroadSynonym>
  <oboInOwl:hasNarrowSynonym>powder XRD</oboInOwl:hasNarrowSynonym>
 </owl:Class>
 <owl:Class rdf:about="{NS}powder"><rdfs:label>powder X-ray diffraction</rdfs:label>
  <rdfs:subClassOf rdf:resource="{NS}xrd"/></owl:Class>
 <owl:Class rdf:about="{NS}neutron"><rdfs:label>neutron diffraction</rdfs:label>
  <rdfs:subClassOf rdf:resource="{FAMILY}"/></owl:Class>
 <owl:Class rdf:about="{NS}old"><rdfs:label>old X-ray diffraction</rdfs:label>
  <rdfs:subClassOf rdf:resource="{FAMILY}"/><owl:deprecated>true</owl:deprecated>
  <obo:IAO_0100001 rdf:resource="{NS}xrd"/></owl:Class>
 <owl:Class rdf:about="{NS}outside"><rdfs:label>small angle scattering</rdfs:label></owl:Class>
 {extra_classes}
</rdf:RDF>'''.encode("utf-8")
    relative = "tests/agent/fixtures/m04-mapping.owl"
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    spec = ResourceSpec(kind="synthetic", path=relative, sha256=sha256(raw),
                        source_url=NS + "source", commit="synthetic-commit", release="synthetic-v1",
                        namespace=NS, license_url=NS + "license", license_name="synthetic-only")
    return load_panet(workspace, spec)


def make_mention(text="X-ray diffraction", context_use="current_study"):
    bundle = synthetic_bundle("Synthetic context: " + text + " was discussed here.\n")
    region = bundle.regions[0]
    start = bundle.text.index(text)
    mention = TechniqueMention(
        original_expression=text,
        mention=evidence_ref(bundle, region.region_id, start=start, end=start + len(text)),
        context=evidence_ref(bundle, region.region_id), context_use=context_use,
    )
    return bundle, mention


@pytest.fixture
def ontology(workspace):
    return synthetic_ontology(workspace)


def decision_for(ontology, text="X-ray diffraction", *, context_use="current_study", limit=50,
                 selected_id=None):
    bundle, mention = make_mention(text, context_use)
    candidates = lexical_candidates(ontology, text, families=(FAMILY,), limit=limit)
    return bundle, propose_mapping(ontology, candidates, bundle, mention, selected_id=selected_id)


@pytest.mark.parametrize("text", ["X-ray diffraction", "XRD", "diffraction by X-rays"])
def test_exact_labels_and_supported_alias_kinds_are_only_unreviewed_proposals(ontology, text):
    bundle, decision = decision_for(ontology, text)
    assert decision.status == "proposed"
    assert decision.proposed_id == NS + "xrd"
    assert decision.approval_status == "unreviewed"
    assert decision.support_status == "not_assessed"
    assert decision.mention.original_expression == text
    assert decision.mention.context_provenance == "supplied_unreviewed"
    assert decision.binding is None
    verify_mapping(ontology, bundle, decision)


@pytest.mark.parametrize("text", ["diffraction probe", "scattering measurement", "powder XRD", "diffraction"])
def test_related_broad_narrow_and_fuzzy_matches_never_assign(ontology, text):
    _, decision = decision_for(ontology, text)
    assert decision.status == "unresolved" and decision.proposed_id is None
    assert decision.reason == "lexical_only"


def test_broad_label_is_not_silently_replaced_by_more_specific_child(ontology):
    _, decision = decision_for(ontology, "X-ray diffraction")
    assert decision.proposed_id == NS + "xrd"
    assert NS + "powder" in {c.concept_id for c in decision.candidates.candidates}
    _, selected_child = decision_for(ontology, "X-ray diffraction", selected_id=NS + "powder")
    assert selected_child.status == "unresolved" and selected_child.reason == "selection_not_exact"


@pytest.mark.parametrize("context,reason", [("related_work", "related_work"), ("uncertain", "usage_uncertain")])
def test_context_annotation_cannot_be_overridden_by_exact_id(ontology, context, reason):
    _, decision = decision_for(ontology, context_use=context, selected_id=NS + "xrd")
    assert decision.status == "unresolved" and decision.reason == reason
    assert decision.proposed_id is None


def test_no_appropriate_concept_is_explicit(ontology):
    _, decision = decision_for(ontology, "unrepresented synthetic technique")
    assert decision.status == "no_match" and decision.reason == "no_lexical_match"
    assert decision.candidates.candidates == ()


@pytest.mark.parametrize("selection,reason", [
    ("not-an-id", "malformed_id"), (NS + "unknown", "unknown_id"),
    (NS + "old", "obsolete_id"), (NS + "outside", "not_retrieved"),
])
def test_invalid_selected_ids_have_distinct_unresolved_states(ontology, selection, reason):
    _, decision = decision_for(ontology, selected_id=selection)
    assert decision.status == "unresolved" and decision.reason == reason
    assert decision.proposed_id is None
    assert decision.selected_id == selection


def test_replacement_link_does_not_resolve_obsolete_selection(ontology):
    obsolete = next(c for c in ontology.concepts if c.concept_id == NS + "old")
    assert obsolete.replaced_by == (NS + "xrd",)
    _, decision = decision_for(ontology, selected_id=obsolete.concept_id)
    assert decision.reason == "obsolete_id" and decision.proposed_id is None


@pytest.mark.parametrize("limit", [1, 50])
def test_ambiguous_exact_aliases_are_unresolved_even_when_top_one_hides_tie(workspace, limit):
    ontology = synthetic_ontology(workspace, extra_classes=f'''
      <owl:Class rdf:about="{NS}ambiguous"><rdfs:label>another technique</rdfs:label>
       <rdfs:subClassOf rdf:resource="{FAMILY}"/>
       <oboInOwl:hasExactSynonym>XRD</oboInOwl:hasExactSynonym></owl:Class>''')
    _, decision = decision_for(ontology, "XRD", limit=limit, selected_id=NS + "ambiguous")
    assert decision.status == "unresolved"
    assert decision.reason == ("candidate_set_truncated" if limit == 1 else "ambiguous_exact_match")


def test_omitted_candidates_after_exact_suffix_block_uniqueness(ontology):
    _, decision = decision_for(ontology, limit=1)
    assert decision.candidates.omitted_candidates > 0
    assert decision.reason == "candidate_set_truncated"


def test_overlap_suffix_proves_omitted_candidates_cannot_hide_an_exact_tie(ontology):
    _, decision = decision_for(ontology, limit=2)
    assert decision.candidates.omitted_candidates > 0
    assert decision.candidates.candidates[-1].match_type == "token_overlap"
    assert decision.status == "proposed" and decision.proposed_id == NS + "xrd"


@pytest.mark.parametrize("target", ["mention", "context"])
def test_quotation_forgery_rejected_even_with_matching_source_hash(ontology, target):
    bundle, mention = make_mention()
    ref = getattr(mention, target).model_copy(update={"quotation": "forged text"})
    mention = mention.model_copy(update={target: ref})
    candidates = lexical_candidates(ontology, "X-ray diffraction", families=(FAMILY,))
    with pytest.raises(ValueError):
        propose_mapping(ontology, candidates, bundle, mention)


def test_context_must_contain_mention_and_original_expression(ontology):
    bundle, mention = make_mention()
    region = bundle.regions[0]
    invalid = evidence_ref(bundle, region.region_id, start=0, end=5)
    with pytest.raises(ValueError, match="contain"):
        TechniqueMention.model_validate({**mention.model_dump(), "context": invalid.model_dump()})
    with pytest.raises(ValueError, match="exact"):
        TechniqueMention.model_validate({**mention.model_dump(), "original_expression": "XRD"})


def test_candidate_query_must_be_exact_original_expression(ontology):
    bundle, mention = make_mention()
    candidates = lexical_candidates(ontology, "XRD", families=(FAMILY,))
    with pytest.raises(ValueError, match="different source expression"):
        propose_mapping(ontology, candidates, bundle, mention)


def test_model_copy_cannot_bypass_resource_or_candidate_validation(ontology):
    bundle, mention = make_mention()
    candidates = lexical_candidates(ontology, "X-ray diffraction", families=(FAMILY,))
    with pytest.raises(ValueError):
        propose_mapping(ontology.model_copy(update={"snapshot_id": "panet:" + "0" * 64}), candidates, bundle, mention)
    tampered = candidates.candidates[0].model_copy(update={"label": "untrusted replacement label"})
    with pytest.raises(ValueError, match="Candidates do not match"):
        propose_mapping(ontology, candidates.model_copy(update={"candidates": (tampered,) + candidates.candidates[1:]}), bundle, mention)


@pytest.fixture
def bound_inputs(workspace, ontology):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    profile = load_profile(workspace)
    record = synthetic_record(profile)
    bundle, mention = make_mention(record["datasets"][0]["measurement_technique"])
    record_snapshot = snapshot(record, bundle, profile)
    candidates = lexical_candidates(ontology, mention.original_expression, families=(FAMILY,), limit=50)
    return bundle, mention, profile, record, record_snapshot, candidates


def test_binding_preserves_canonical_technique_and_stable_dataset_handle(ontology, bound_inputs):
    bundle, mention, _, record, record_snapshot, candidates = bound_inputs
    before = deepcopy(record_snapshot.model_dump())
    decision = propose_mapping(ontology, candidates, bundle, mention, record=record_snapshot,
                               assertion_path="/datasets/0/measurement_technique")
    dataset = next(e for e in record_snapshot.entities if e.local_id == "dataset_1")
    assert decision.binding.entity_handle == dataset.handle
    assert decision.binding.revision_id == record_snapshot.revision_id
    assert decision.binding.value_json == '"X-ray diffraction"'
    assert record_snapshot.model_dump() == before
    assert record_snapshot.canonical == record
    verify_mapping(ontology, bundle, decision, record_snapshot)


@pytest.mark.parametrize("path", ["/bibliographic/title", "/datasets/00/measurement_technique",
                                   "/datasets/0/dataset_name", "/datasets/9/measurement_technique"])
def test_binding_only_targets_existing_measurement_technique(ontology, bound_inputs, path):
    bundle, mention, _, _, record_snapshot, candidates = bound_inputs
    with pytest.raises(ValueError):
        propose_mapping(ontology, candidates, bundle, mention, record=record_snapshot, assertion_path=path)


def test_record_and_path_are_required_together(ontology, bound_inputs):
    bundle, mention, _, _, record_snapshot, candidates = bound_inputs
    for args in ({"record": record_snapshot}, {"assertion_path": "/datasets/0/measurement_technique"}):
        with pytest.raises(ValueError, match="both"):
            propose_mapping(ontology, candidates, bundle, mention, **args)


def test_record_binding_rejects_wrong_value_and_wrong_source(ontology, bound_inputs):
    bundle, mention, profile, record, record_snapshot, candidates = bound_inputs
    record["datasets"][0]["measurement_technique"] = "changed wording"
    changed = snapshot(record, bundle, profile)
    with pytest.raises(ValueError, match="exact original"):
        propose_mapping(ontology, candidates, bundle, mention, record=changed,
                        assertion_path="/datasets/0/measurement_technique")
    other_bundle = synthetic_bundle(bundle.text + "Another source.\n")
    changed = snapshot(record_snapshot.canonical, other_bundle, profile)
    with pytest.raises(ValueError, match="different source"):
        propose_mapping(ontology, candidates, bundle, mention, record=changed,
                        assertion_path="/datasets/0/measurement_technique")


def test_replay_rejects_stale_record_revision_even_if_technique_is_unchanged(ontology, bound_inputs):
    bundle, mention, profile, record, record_snapshot, candidates = bound_inputs
    decision = propose_mapping(ontology, candidates, bundle, mention, record=record_snapshot,
                               assertion_path="/datasets/0/measurement_technique")
    record["datasets"][0]["dataset_name"] = "Changed dataset name"
    changed = snapshot(record, bundle, profile)
    with pytest.raises(ValueError, match="no longer matches"):
        verify_mapping(ontology, bundle, decision, changed)


def test_mapping_has_no_approval_or_supported_state(ontology):
    _, decision = decision_for(ontology)
    for field, value in [("approval_status", "approved"), ("support_status", "supported"), ("status", "accepted")]:
        with pytest.raises(ValidationError):
            MappingDecision.model_validate({**decision.model_dump(), field: value})


def test_replay_recomputes_semantics_even_if_tamperer_recomputes_digest(ontology):
    bundle, decision = decision_for(ontology, context_use="related_work")
    payload = decision.model_dump(mode="json", exclude={"cache_key"})
    payload.update(status="proposed", reason="exact_lexical_candidate", proposed_id=NS + "xrd")
    forged = MappingDecision.model_validate({**payload, "cache_key": "mapping:" + digest(payload)})
    with pytest.raises(ValueError, match="no longer matches"):
        verify_mapping(ontology, bundle, forged)


def test_cache_identity_changes_with_context_and_retained_expression(ontology):
    _, current = decision_for(ontology)
    _, related = decision_for(ontology, context_use="related_work")
    _, alias = decision_for(ontology, "XRD")
    assert len({current.cache_key, related.cache_key, alias.cache_key}) == 3
    with pytest.raises(ValueError, match="cache identity"):
        MappingDecision.model_validate({**current.model_dump(), "cache_key": related.cache_key})


def test_replay_resolves_source_again_and_rejects_changed_bundle(ontology):
    _, decision = decision_for(ontology)
    changed, _ = make_mention("X-ray diffraction with changed context")
    with pytest.raises(ValueError):
        verify_mapping(ontology, changed, decision)
