"""Synthetic lexical checks, not expert-reviewed PaNET mapping measurements."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.panet_lookup import (
    CandidateSet, lexical_candidates, normalize_query, verify_candidates,
)
from meta_data_extraction.agent.panet_resources import ResourceSpec, load_panet


NS = "urn:synthetic:"
DIFFRACTION = NS + "diffraction"
SCATTERING = NS + "scattering"
XRD = NS + "xrd"
POWDER = NS + "powder"
SAXS = NS + "saxs"
RETIRED = NS + "retired"
OUTSIDE = NS + "outside"

RDF = '''<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#"
 xmlns:skos="http://www.w3.org/2004/02/skos/core#"
 xmlns:obo="http://purl.obolibrary.org/obo/"
 xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
 <owl:Ontology rdf:about="urn:synthetic:ontology"><owl:versionInfo>fixture-v1</owl:versionInfo></owl:Ontology>
 <owl:Class rdf:about="urn:synthetic:diffraction"><rdfs:label>diffraction</rdfs:label></owl:Class>
 <owl:Class rdf:about="urn:synthetic:scattering"><rdfs:label>scattering</rdfs:label></owl:Class>
 <owl:Class rdf:about="urn:synthetic:xrd">
  <rdfs:label>X-ray diffraction</rdfs:label>
  <rdfs:subClassOf rdf:resource="urn:synthetic:diffraction"/>
  <skos:altLabel>XRD</skos:altLabel>
  <oboInOwl:hasExactSynonym>synthetic exact diffraction alias</oboInOwl:hasExactSynonym>
  <oboInOwl:hasRelatedSynonym>synthetic related alias</oboInOwl:hasRelatedSynonym>
  <oboInOwl:hasBroadSynonym>synthetic broad alias</oboInOwl:hasBroadSynonym>
  <oboInOwl:hasNarrowSynonym>synthetic narrow alias</oboInOwl:hasNarrowSynonym>
 </owl:Class>
 <owl:Class rdf:about="urn:synthetic:powder">
  <rdfs:label>powder X-ray diffraction</rdfs:label>
  <rdfs:subClassOf rdf:resource="urn:synthetic:xrd"/>
 </owl:Class>
 <owl:Class rdf:about="urn:synthetic:saxs">
  <rdfs:label>small-angle X-ray scattering</rdfs:label>
  <rdfs:subClassOf rdf:resource="urn:synthetic:scattering"/>
  <skos:altLabel>SAXS</skos:altLabel>
 </owl:Class>
 <owl:Class rdf:about="urn:synthetic:retired">
  <rdfs:label>retired experimental method</rdfs:label>
  <rdfs:subClassOf rdf:resource="urn:synthetic:diffraction"/>
  <owl:deprecated>true</owl:deprecated>
  <obo:IAO_0100001 rdf:resource="urn:synthetic:xrd"/>
 </owl:Class>
 <owl:Class rdf:about="urn:synthetic:outside"><rdfs:label>unrelated microscopy</rdfs:label></owl:Class>
</rdf:RDF>'''


def snapshot_at(workspace: Path, *, text: str = RDF, release: str = "fixture-v1"):
    relative = "tests/agent/fixtures/panet_lookup.owl"
    raw = text.encode("utf-8")
    (workspace / relative).write_bytes(raw)
    spec = ResourceSpec(
        kind="synthetic", path=relative, sha256=sha256(raw),
        source_url="urn:synthetic:fixture", commit="synthetic", release=release,
        namespace=NS, license_url="urn:synthetic:license", license_name="synthetic fixture",
    )
    return load_panet(workspace, spec)


@pytest.fixture
def ontology(workspace):
    return snapshot_at(workspace)


def lookup(ontology, query, **kwargs):
    return lexical_candidates(ontology, query, families=(DIFFRACTION, SCATTERING), **kwargs)


def test_original_expression_preserved_and_known_label_matches(ontology):
    query = "  Ｘ–ＲＡＹ  diffraction  "
    result = lookup(ontology, query)
    best = result.candidates[0]
    assert result.query == query
    assert result.normalized_query == "x ray diffraction"
    assert best.concept_id == XRD and best.label == "X-ray diffraction"
    assert best.matched_text == "X-ray diffraction"
    assert best.match_type == "exact" and best.matched_kind == "label" and best.score == 1
    assert result.resource_kind == "synthetic"
    assert verify_candidates(ontology, result) == result


@pytest.mark.parametrize("query,kind", [
    ("XRD", "alternative"), ("synthetic exact diffraction alias", "exact"),
    ("synthetic related alias", "related"), ("synthetic broad alias", "broad"),
    ("synthetic narrow alias", "narrow"),
])
def test_supplied_synonym_kind_retained_without_semantic_equivalence(ontology, query, kind):
    result = lookup(ontology, query)
    best = result.candidates[0]
    assert best.concept_id == XRD and best.match_type == "exact"
    assert best.matched_kind == kind
    assert not hasattr(best, "approved") and not hasattr(best, "confidence")


def test_broad_label_outranks_specific_children_without_collapsing_them(ontology):
    result = lookup(ontology, "diffraction")
    assert result.candidates[0].concept_id == DIFFRACTION
    assert result.candidates[0].match_type == "exact"
    assert {candidate.concept_id for candidate in result.candidates} >= {DIFFRACTION, XRD, POWDER}
    assert next(candidate for candidate in result.candidates if candidate.concept_id == POWDER).match_type == "token_overlap"
    assert next(candidate for candidate in result.candidates if candidate.concept_id == POWDER).parents == (XRD,)


def test_alias_not_inherited_by_descendants_and_no_acronyms_invented(ontology):
    assert [candidate.concept_id for candidate in lookup(ontology, "XRD").candidates] == [XRD]
    assert lookup(ontology, "PXRD").status == "no_match"
    assert lookup(ontology, "SAXS").candidates[0].concept_id == SAXS


def test_only_explicit_families_are_searchable(ontology):
    assert lexical_candidates(ontology, "SAXS", families=(DIFFRACTION,)).status == "no_match"
    assert lookup(ontology, "unrelated microscopy").status == "no_match"
    outside = lexical_candidates(ontology, "unrelated microscopy", families=(OUTSIDE,))
    assert outside.candidates[0].concept_id == OUTSIDE


def test_obsolete_labels_and_replacements_are_not_automatically_selected(ontology):
    result = lookup(ontology, "retired experimental method")
    assert result.status == "no_match" and result.candidates == ()
    assert all(candidate.concept_id != RETIRED for candidate in lookup(ontology, "diffraction").candidates)


def test_overlap_is_token_dice_and_reordered_words_are_not_exact(ontology):
    result = lookup(ontology, "ray diffraction X")
    best = result.candidates[0]
    assert best.concept_id == XRD and best.score == 1
    assert best.match_type == "token_overlap"
    broad = next(candidate for candidate in result.candidates if candidate.concept_id == DIFFRACTION)
    assert broad.score == 0.5


def test_limits_and_canonical_scope_order_are_deterministic(ontology):
    a = lookup(ontology, "diffraction", limit=1)
    b = lexical_candidates(ontology, "diffraction", families=(SCATTERING, DIFFRACTION), limit=1)
    assert a == b and len(a.candidates) == 1
    assert a.total_candidates == 3 and a.omitted_candidates == 2
    assert a.candidates[0].concept_id == DIFFRACTION


def test_ties_use_concept_id_not_resource_iteration(workspace):
    text = RDF.replace("</rdf:RDF>", '''<owl:Class rdf:about="urn:synthetic:tie_a">
       <rdfs:label>neutron scattering</rdfs:label><rdfs:subClassOf rdf:resource="urn:synthetic:scattering"/>
       </owl:Class><owl:Class rdf:about="urn:synthetic:tie_b"><rdfs:label>optical scattering</rdfs:label>
       <rdfs:subClassOf rdf:resource="urn:synthetic:scattering"/></owl:Class></rdf:RDF>''')
    result = lookup(snapshot_at(workspace, text=text), "scattering")
    assert [candidate.concept_id for candidate in result.candidates[1:3]] == [NS + "tie_a", NS + "tie_b"]


@pytest.mark.parametrize("query", ["", "!!!", "___", " " * 12, "a" * 4097, None, 42])
def test_invalid_queries_fail_explicitly(ontology, query):
    with pytest.raises(ValueError):
        lookup(ontology, query)


@pytest.mark.parametrize("limit", [0, -1, 51, True, 1.0, None])
def test_invalid_limits_fail_explicitly(ontology, limit):
    with pytest.raises(ValueError):
        lookup(ontology, "XRD", limit=limit)


@pytest.mark.parametrize("families", [(), (DIFFRACTION, DIFFRACTION), (RETIRED,), (NS + "unknown",), [DIFFRACTION], (1,)])
def test_invalid_families_fail_explicitly(ontology, families):
    with pytest.raises(ValueError):
        lexical_candidates(ontology, "XRD", families=families)


def test_no_match_is_explicit_not_an_empty_success(ontology):
    result = lookup(ontology, "unrepresentedtechnique")
    assert result.status == "no_match" and result.candidates == ()
    assert result.total_candidates == result.omitted_candidates == 0
    assert verify_candidates(ontology, result) == result


def test_result_roundtrip_is_immutable_and_revalidates_structure(ontology):
    result = lookup(ontology, "XRD")
    assert CandidateSet.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        result.candidates[0].concept_id = OUTSIDE
    changed = result.model_copy(update={"normalized_query": "other"})
    with pytest.raises(ValueError):
        verify_candidates(ontology, changed)


@pytest.mark.parametrize("updates", [
    {"concept_id": NS + "unknown"}, {"concept_id": RETIRED}, {"label": "fabricated"},
    {"parents": (OUTSIDE,)}, {"matched_kind": "exact"}, {"score": 0.25},
])
def test_replay_cannot_inject_or_modify_candidates(ontology, updates):
    result = lookup(ontology, "XRD")
    candidate = result.candidates[0].model_copy(update=updates)
    tampered = result.model_copy(update={"candidates": (candidate,)})
    with pytest.raises(ValueError):
        verify_candidates(ontology, tampered)


def test_snapshot_model_copy_cannot_bypass_integrity(ontology):
    changed_concept = ontology.concepts[0].model_copy(update={"label": "tampered"})
    tampered = ontology.model_copy(update={"concepts": (changed_concept, *ontology.concepts[1:])})
    with pytest.raises(ValueError):
        lookup(tampered, "XRD")


def test_cache_invalidation_covers_resource_version_content_scope_query_and_limit(workspace, ontology):
    baseline = lookup(ontology, "XRD")
    variants = [
        lookup(snapshot_at(workspace, release="fixture-v2"), "XRD"),
        lookup(snapshot_at(workspace, text=RDF.replace("fixture-v1", "fixture-v2")), "XRD"),
        lexical_candidates(ontology, "XRD", families=(DIFFRACTION,)),
        lookup(ontology, "xrd"), lookup(ontology, "XRD", limit=1),
    ]
    assert all(result.cache_key != baseline.cache_key for result in variants)
    changed = snapshot_at(workspace, release="fixture-v2")
    with pytest.raises(ValueError):
        verify_candidates(changed, baseline)


def test_lookup_never_reads_repository_or_source_files(ontology, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Lookup must use only the supplied immutable snapshot")
    monkeypatch.setattr(Path, "read_bytes", denied)
    monkeypatch.setattr(Path, "read_text", denied)
    result = lookup(ontology, "XRD")
    assert verify_candidates(ontology, result) == result


def test_normalization_does_not_stem_or_add_synonyms():
    assert normalize_query("X-ray_diffraction") == "x ray diffraction"
    assert normalize_query("diffractions") != normalize_query("diffraction")
