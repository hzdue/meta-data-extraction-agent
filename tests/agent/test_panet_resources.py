"""Offline resource boundary tests use only clearly synthetic namespaces."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.panet_resources import (
    MAX_RESOURCE_BYTES, Concept, OntologySnapshot, ResourceError, ResourceSpec,
    Synonym, load_panet, verify_resources,
)

HEADER = '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#"
 xmlns:skos="http://www.w3.org/2004/02/skos/core#"
 xmlns:obo="http://purl.obolibrary.org/obo/"
 xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
<owl:Ontology rdf:about="urn:synthetic:ontology"><owl:versionInfo>fixture-v1</owl:versionInfo>{metadata}</owl:Ontology>
'''
CLASSES = '''
<owl:Class rdf:about="urn:synthetic:parent"><rdfs:label>diffraction</rdfs:label></owl:Class>
<owl:Class rdf:about="urn:synthetic:child">
 <rdfs:label>test diffraction</rdfs:label>
 <rdfs:subClassOf rdf:resource="urn:synthetic:parent"/>
 <skos:altLabel>TD</skos:altLabel>
 <oboInOwl:hasExactSynonym>exact test diffraction</oboInOwl:hasExactSynonym>
 <obo:IAO_0000115>A synthetic test definition.</obo:IAO_0000115>
</owl:Class>
'''


def resource(tmp_path: Path, *, classes: str = CLASSES, metadata: str = "", raw: bytes | None = None):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src/meta_data_extraction").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "tests/agent/fixtures/panet.owl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else (HEADER.replace("{metadata}", metadata) + classes + "</rdf:RDF>").encode()
    path.write_bytes(payload)
    spec = ResourceSpec(kind="synthetic", path="tests/agent/fixtures/panet.owl", sha256=sha256(payload),
                        source_url="urn:synthetic:fixture", commit="synthetic", release="fixture-v1",
                        namespace="urn:synthetic:", license_url="urn:synthetic:license", license_name="synthetic fixture")
    return spec, path


def test_load_labels_declared_alias_kinds_named_parents_and_definitions(tmp_path):
    spec, _ = resource(tmp_path)
    snapshot = load_panet(tmp_path, spec)
    child, parent = snapshot.concepts
    assert child.concept_id == "urn:synthetic:child"
    assert child.parents == (parent.concept_id,)
    assert child.definition == "A synthetic test definition."
    assert [(s.text, s.kind) for s in child.synonyms] == [("TD", "alternative"), ("exact test diffraction", "exact")]
    assert child.synonyms[0].predicate == "http://www.w3.org/2004/02/skos/core#altLabel"
    assert snapshot.ontology_version == ("fixture-v1",)
    verify_resources(snapshot, tmp_path)
    assert OntologySnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_changed_bytes_fail_before_parsing(tmp_path):
    spec, path = resource(tmp_path)
    snapshot = load_panet(tmp_path, spec)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ResourceError, match="SHA-256"):
        verify_resources(snapshot, tmp_path)


def test_different_resource_and_release_produce_distinct_snapshot_ids(tmp_path):
    spec, _ = resource(tmp_path)
    first = load_panet(tmp_path, spec)
    release_changed = ResourceSpec.model_validate({**spec.model_dump(), "release": "fixture-v2"})
    assert load_panet(tmp_path, release_changed).snapshot_id != first.snapshot_id
    spec2, _ = resource(tmp_path, classes=CLASSES.replace("test diffraction", "changed diffraction"))
    assert load_panet(tmp_path, spec2).snapshot_id != first.snapshot_id


def test_snapshot_cannot_silently_change_concepts(tmp_path):
    spec, _ = resource(tmp_path)
    payload = load_panet(tmp_path, spec).model_dump(mode="json")
    payload["concepts"][0]["label"] = "tampered"
    with pytest.raises(ValidationError, match="identity"):
        OntologySnapshot.model_validate(payload)


def test_failed_pin_and_unavailable_resource_are_explicit(tmp_path):
    spec, path = resource(tmp_path)
    with pytest.raises(ResourceError, match="SHA-256"):
        load_panet(tmp_path, ResourceSpec.model_validate({**spec.model_dump(), "sha256": "0" * 64}))
    path.rename(path.with_suffix(".hidden"))
    with pytest.raises(ResourceError, match="unavailable"):
        load_panet(tmp_path, spec)


@pytest.mark.parametrize("raw", [b"<not-rdf/>", b"bad", b"<rdf:RDF", b"\xff\xfe", b"\x00<rdf:RDF/>"])
def test_malformed_and_non_utf8_formats_are_rejected(tmp_path, raw):
    spec, _ = resource(tmp_path, raw=raw)
    with pytest.raises(ResourceError):
        load_panet(tmp_path, spec)


@pytest.mark.parametrize("declaration", [
    '<!DOCTYPE rdf:RDF [<!ENTITY payload "expanded">]>',
    '<!ENTITY file SYSTEM "file:///private">',
    '<!DOCTYPE rdf:RDF SYSTEM "https://invalid.example/dtd">',
])
def test_entities_and_dtd_never_expand_or_download(tmp_path, declaration):
    raw = (HEADER.replace("{metadata}", "").replace('<?xml version="1.0"?>', '<?xml version="1.0"?>' + declaration)
           + CLASSES + "</rdf:RDF>").encode()
    spec, _ = resource(tmp_path, raw=raw)
    with pytest.raises(ResourceError, match="DTD"):
        load_panet(tmp_path, spec)


def test_size_bound_precedes_parse(tmp_path):
    spec, _ = resource(tmp_path, raw=b" " * (MAX_RESOURCE_BYTES + 1))
    with pytest.raises(ResourceError, match="size"):
        load_panet(tmp_path, spec)


def test_imports_and_complex_axioms_are_described_without_expansion(tmp_path):
    complex_classes = CLASSES.replace('<rdfs:subClassOf rdf:resource="urn:synthetic:parent"/>', '''
       <rdfs:subClassOf><owl:Restriction><owl:onProperty rdf:resource="urn:synthetic:probe"/>
       <owl:someValuesFrom rdf:resource="urn:synthetic:parent"/></owl:Restriction></rdfs:subClassOf>''')
    spec, _ = resource(tmp_path, classes=complex_classes, metadata='<owl:imports rdf:resource="https://invalid.example/resource.owl"/>')
    snapshot = load_panet(tmp_path, spec)
    assert snapshot.concepts[0].parents == ()
    assert any("imports were not fetched" in line for line in snapshot.limitations)
    assert any("Complex OWL" in line for line in snapshot.limitations)


def test_obsolete_flag_and_replacement_do_not_invent_mapping(tmp_path):
    classes = CLASSES.replace('<rdfs:label>test diffraction</rdfs:label>', '''<rdfs:label>test diffraction</rdfs:label>
       <owl:deprecated>true</owl:deprecated><obo:IAO_0100001 rdf:resource="urn:synthetic:parent"/>''')
    spec, _ = resource(tmp_path, classes=classes)
    child = load_panet(tmp_path, spec).concepts[0]
    assert child.obsolete is True
    assert child.replaced_by == ("urn:synthetic:parent",)
    assert child.concept_id == "urn:synthetic:child"


@pytest.mark.parametrize("annotation", ["<owl:deprecated>maybe</owl:deprecated>", "<owl:deprecated>true</owl:deprecated><owl:deprecated>false</owl:deprecated>"])
def test_invalid_obsolete_annotations_are_errors(tmp_path, annotation):
    spec, _ = resource(tmp_path, classes=CLASSES.replace("<rdfs:label>test diffraction</rdfs:label>", "<rdfs:label>test diffraction</rdfs:label>" + annotation))
    with pytest.raises(ResourceError, match="obsolete"):
        load_panet(tmp_path, spec)


def test_unknown_external_namespace_not_in_candidate_index(tmp_path):
    classes = CLASSES + '<owl:Class rdf:about="https://external.example/not-a-technique"><rdfs:label>external</rdfs:label></owl:Class>'
    classes = classes.replace('<rdfs:subClassOf rdf:resource="urn:synthetic:parent"/>', '<rdfs:subClassOf rdf:resource="https://external.example/parent"/>')
    spec, _ = resource(tmp_path, classes=classes)
    snapshot = load_panet(tmp_path, spec)
    assert len(snapshot.concepts) == 2
    assert snapshot.concepts[0].parents == ("https://external.example/parent",)
    assert any("external" in line for line in snapshot.limitations)


@pytest.mark.parametrize("classes", [
    CLASSES + CLASSES,
    CLASSES.replace("<rdfs:label>test diffraction</rdfs:label>", ""),
    CLASSES.replace("<rdfs:label>test diffraction</rdfs:label>", "<rdfs:label>one</rdfs:label><rdfs:label>two</rdfs:label>"),
    CLASSES.replace('rdf:resource="urn:synthetic:parent"', 'rdf:resource="relative"'),
    CLASSES.replace('<owl:Class rdf:about="urn:synthetic:child">', '<owl:Class xml:base="https://invalid.example/" rdf:about="urn:synthetic:child">'),
    "",
])
def test_unsupported_or_ambiguous_projections_fail_explicitly(tmp_path, classes):
    spec, _ = resource(tmp_path, classes=classes)
    with pytest.raises(ResourceError):
        load_panet(tmp_path, spec)


def test_language_preserved_by_explicit_scope_limitation(tmp_path):
    classes = CLASSES.replace("<rdfs:label>test diffraction</rdfs:label>", '<rdfs:label xml:lang="en">test diffraction</rdfs:label><rdfs:label xml:lang="de">Testbeugung</rdfs:label>')
    spec, _ = resource(tmp_path, classes=classes)
    snapshot = load_panet(tmp_path, spec)
    assert snapshot.concepts[0].label == "test diffraction"
    assert any("English" in line for line in snapshot.limitations)


def test_named_rdf_description_class_is_supported(tmp_path):
    classes = CLASSES.replace('<owl:Class rdf:about="urn:synthetic:child">', '<rdf:Description rdf:about="urn:synthetic:child"><rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Class"/>')
    classes = classes.rsplit("</owl:Class>", 1)[0] + "</rdf:Description>"
    spec, _ = resource(tmp_path, classes=classes)
    assert len(load_panet(tmp_path, spec).concepts) == 2


@pytest.mark.parametrize("changes", [
    {"path": "data/gold/example.owl"}, {"path": "runs/agent/fake/panet.owl"},
    {"namespace": "http://purl.org/pan-science/PaNET/"}, {"source_url": "https://real.example/"},
])
def test_synthetic_namespace_and_fixture_location_cannot_be_disguised(tmp_path, changes):
    spec, _ = resource(tmp_path)
    with pytest.raises(ValidationError):
        ResourceSpec.model_validate({**spec.model_dump(), **changes})


@pytest.mark.parametrize("path", ["tests/agent/fixtures/../../gold/file.owl", "tests/agent/fixtures/C:bad.owl"])
def test_path_policy_reused(tmp_path, path):
    spec, _ = resource(tmp_path)
    with pytest.raises(ResourceError):
        load_panet(tmp_path, ResourceSpec.model_validate({**spec.model_dump(), "path": path}))


def test_immutable_typed_concepts_and_synonyms():
    concept = Concept(concept_id="urn:synthetic:x", label="x", synonyms=(Synonym(text="y", kind="related"),))
    with pytest.raises(ValidationError):
        concept.label = "changed"
    with pytest.raises(ValidationError):
        Synonym(text="", kind="exact")
    with pytest.raises(ValidationError):
        Concept(concept_id="urn:synthetic:x", label="x", obsolete="false")
