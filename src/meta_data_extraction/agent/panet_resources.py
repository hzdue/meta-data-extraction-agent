"""Pinned, local PaNET RDF/XML projection; no imports, reasoner, or network I/O.

Only explicit named-class annotations and direct named parents are projected.
This deliberately does not interpret OWL restrictions or equivalence axioms.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from pydantic import Field, model_validator

from .contracts import digest
from .evidence import FrozenModel, sha256
from .paths import PathPolicyError, input_file

PARSER_VERSION = "m04-panet-rdfxml-v1"
PANET_NAMESPACE = "http://purl.org/pan-science/PaNET/"
MAX_RESOURCE_BYTES = 4_000_000
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
OBO = "http://purl.obolibrary.org/obo/"
OBOINOWL = "http://www.geneontology.org/formats/oboInOwl#"
XML = "http://www.w3.org/XML/1998/namespace"


class ResourceError(ValueError):
    """A standards resource is unavailable, malformed, or differs from its pin."""


class ResourceSpec(FrozenModel):
    kind: Literal["authentic", "synthetic"]
    path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_url: str = Field(min_length=1, max_length=1500)
    commit: str = Field(min_length=1, max_length=100)
    release: str = Field(min_length=1, max_length=100)
    namespace: str = Field(min_length=1, max_length=300)
    license_url: str = Field(min_length=1, max_length=500)
    license_name: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _provenance(self) -> ResourceSpec:
        path = self.path.replace("\\", "/")
        if self.kind == "synthetic":
            if not path.startswith("tests/agent/fixtures/") or not self.namespace.startswith("urn:synthetic:"):
                raise ValueError("Synthetic resources require the test-fixture path and namespace")
            if not self.source_url.startswith("urn:synthetic:"):
                raise ValueError("Synthetic provenance must be explicitly synthetic")
        else:
            if self.namespace != PANET_NAMESPACE or not re.fullmatch(r"[a-f0-9]{40}", self.commit):
                raise ValueError("Authentic PaNET requires its namespace and an exact commit")
            url = urlsplit(self.source_url)
            prefix = f"/pan-ontologies/PaNET/{self.commit}/"
            if (url.scheme != "https" or url.netloc != "raw.githubusercontent.com"
                    or not url.path.startswith(prefix) or url.query or url.fragment):
                raise ValueError("Authentic source URL must pin an official repository file")
            if not path.startswith("runs/agent/"):
                raise ValueError("Authentic snapshots belong in an isolated agent run")
            if (self.license_url != "https://creativecommons.org/licenses/by/4.0/"
                    or self.license_name != "CC-BY-4.0"):
                raise ValueError("Record the verified PaNET resource license")
        return self


class Synonym(FrozenModel):
    text: str = Field(min_length=1, max_length=2000)
    kind: Literal["exact", "related", "broad", "narrow", "alternative"]
    predicate: str | None = None


class Concept(FrozenModel):
    concept_id: str = Field(min_length=1, max_length=500)
    label: str = Field(min_length=1, max_length=2000)
    synonyms: tuple[Synonym, ...] = ()
    parents: tuple[str, ...] = ()
    obsolete: bool = Field(default=False, strict=True)
    definition: str | None = None
    replaced_by: tuple[str, ...] = ()


class OntologySnapshot(FrozenModel):
    resource: ResourceSpec
    snapshot_id: str = Field(pattern=r"^panet:[a-f0-9]{64}$")
    concepts: tuple[Concept, ...] = Field(min_length=1, max_length=10000)
    limitations: tuple[str, ...] = ()
    parser_version: Literal["m04-panet-rdfxml-v1"] = PARSER_VERSION
    ontology_version: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _integrity(self) -> OntologySnapshot:
        ids = [concept.concept_id for concept in self.concepts]
        if ids != sorted(set(ids)) or any(not _in_namespace(i, self.resource) for i in ids):
            raise ValueError("Snapshot concepts must be unique, sorted, and in the declared namespace")
        payload = self.model_dump(mode="json", exclude={"snapshot_id"})
        if self.snapshot_id != "panet:" + digest(payload):
            raise ValueError("Ontology snapshot identity does not match its content")
        return self


def _in_namespace(iri: str, spec: ResourceSpec) -> bool:
    if spec.kind == "authentic":
        return re.fullmatch(re.escape(PANET_NAMESPACE) + r"PaNET[0-9]+", iri) is not None
    return iri.startswith(spec.namespace) and len(iri) > len(spec.namespace)


def _tag(namespace: str, name: str) -> str:
    return "{" + namespace + "}" + name


def _texts(element: ET.Element, tag: str) -> tuple[str, ...]:
    # The chosen released ontology has English/unqualified literal annotations.
    # Other-language literals are not silently treated as English synonyms.
    return tuple(sorted({(child.text or "").strip() for child in element.findall(tag)
                         if child.get(_tag(XML, "lang"), "en").lower() in {"en", ""}
                         and (child.text or "").strip()}))


_SYNONYM_PREDICATES = {
    _tag(SKOS, "altLabel"): ("alternative", SKOS + "altLabel"),
    **{_tag(OBOINOWL, "has" + name + "Synonym"): (name.lower(), OBOINOWL + "has" + name + "Synonym")
       for name in ("Exact", "Related", "Broad", "Narrow")},
}


def _parse(raw: bytes, spec: ResourceSpec) -> OntologySnapshot:
    if len(raw) > MAX_RESOURCE_BYTES:
        raise ResourceError("Resource exceeds the bounded RDF/XML size")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ResourceError("Resource must be UTF-8 RDF/XML") from exc
    if "\x00" in text or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.I):
        raise ResourceError("DTD and entity declarations are not permitted")
    try:
        document = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ResourceError("Malformed RDF/XML resource") from exc
    if document.tag != _tag(RDF, "RDF"):
        raise ResourceError("Expected an RDF/XML root")
    if any(_tag(XML, "base") in node.attrib for node in document.iter() if node is not document):
        raise ResourceError("Nested XML base changes are not supported")
    limitations = {"Only explicit named parents are loaded; no OWL reasoning or equivalence expansion."}
    if document.findall(".//" + _tag(OWL, "imports")):
        limitations.add("Declared ontology imports were not fetched or interpreted.")
    if any(node.tag in {_tag(OWL, name) for name in ("equivalentClass", "intersectionOf", "Restriction", "unionOf")}
           for node in document.iter()):
        limitations.add("Complex OWL class axioms are present but are not projected into parent links.")
    if any(node.get(_tag(XML, "lang"), "en").lower() not in {"en", ""} for node in document.iter()):
        limitations.add("Only English and unqualified literal labels and annotations are indexed.")
    ontology = document.findall(_tag(OWL, "Ontology"))
    if len(ontology) != 1:
        raise ResourceError("Expected exactly one ontology declaration")
    license_iris = {n.get(_tag(RDF, "resource")) for n in ontology[0].findall(_tag("http://purl.org/dc/terms/", "license"))}
    if spec.kind == "authentic" and spec.license_url not in license_iris:
        raise ResourceError("Ontology metadata does not confirm the pinned license")
    concepts: list[Concept] = []
    for element in document:
        typed_class = (element.tag == _tag(OWL, "Class") or
                       element.tag == _tag(RDF, "Description") and any(
                           t.get(_tag(RDF, "resource")) == OWL + "Class"
                           for t in element.findall(_tag(RDF, "type"))))
        iri = element.get(_tag(RDF, "about"), "")
        if not typed_class or not _in_namespace(iri, spec):
            continue
        labels = _texts(element, _tag(RDFS, "label"))
        if len(labels) != 1:
            raise ResourceError("Each selected class needs exactly one English or unqualified label")
        synonyms = [Synonym(text=value, kind=kind, predicate=predicate)
                    for tag, (kind, predicate) in _SYNONYM_PREDICATES.items()
                    for value in _texts(element, tag)]
        parents = tuple(sorted({p.get(_tag(RDF, "resource")) for p in element.findall(_tag(RDFS, "subClassOf"))
                                if p.get(_tag(RDF, "resource"))}))
        if any(not re.match(r"^(?:https?://|urn:)", parent) for parent in parents):
            raise ResourceError("Relative parent IRIs are not supported")
        obsolete_values = _texts(element, _tag(OWL, "deprecated"))
        if any(value not in {"true", "false", "1", "0"} for value in obsolete_values):
            raise ResourceError("Invalid explicit obsolete flag")
        if any(v in {"true", "1"} for v in obsolete_values) and any(v in {"false", "0"} for v in obsolete_values):
            raise ResourceError("Conflicting obsolete flags")
        definitions = _texts(element, _tag(OBO, "IAO_0000115"))
        replaced_by = tuple(sorted({n.get(_tag(RDF, "resource")) for n in element.findall(_tag(OBO, "IAO_0100001"))
                                    if n.get(_tag(RDF, "resource"))}))
        concepts.append(Concept(concept_id=iri, label=labels[0], synonyms=tuple(sorted(synonyms, key=lambda s: (s.text, s.kind))),
                                parents=parents, obsolete=any(v in {"true", "1"} for v in obsolete_values),
                                definition="\n\n".join(definitions) or None, replaced_by=replaced_by))
    if not concepts:
        raise ResourceError("No labeled concepts in the approved namespace")
    ids = [c.concept_id for c in concepts]
    if len(set(ids)) != len(ids):
        raise ResourceError("Duplicate selected class declarations")
    known = set(ids)
    if any(parent not in known for c in concepts for parent in c.parents):
        limitations.add("Some named parents are external or lack a selected declaration; their definitions are not loaded.")
    payload = {"resource": spec.model_dump(mode="json"), "concepts": [c.model_dump(mode="json") for c in sorted(concepts, key=lambda c: c.concept_id)],
               "limitations": sorted(limitations), "parser_version": PARSER_VERSION,
               "ontology_version": list(_texts(ontology[0], _tag(OWL, "versionInfo")))}
    return OntologySnapshot.model_validate({**payload, "snapshot_id": "panet:" + digest(payload)})


def load_panet(root: Path | str, spec: ResourceSpec) -> OntologySnapshot:
    """Load one explicitly pinned resource. No directory scanning or downloading."""
    spec = ResourceSpec.model_validate(spec.model_dump())
    allowed = ("runs/agent",) if spec.kind == "authentic" else ("tests/agent/fixtures",)
    try:
        path = input_file(root, spec.path, allowed_roots=allowed)
        if path.stat().st_size > MAX_RESOURCE_BYTES:
            raise ResourceError("Resource exceeds the bounded RDF/XML size")
        raw = path.read_bytes()
    except (OSError, PathPolicyError) as exc:
        raise ResourceError("Pinned ontology resource is unavailable at its approved path") from exc
    if sha256(raw) != spec.sha256:
        raise ResourceError("Ontology bytes do not match the pinned SHA-256")
    return _parse(raw, spec)


def verify_resources(snapshot: OntologySnapshot, root: Path | str) -> None:
    """Reject changed resources, parser projections, or serialized snapshots."""
    expected = OntologySnapshot.model_validate(snapshot.model_dump())
    if load_panet(root, expected.resource) != expected:
        raise ResourceError("Ontology snapshot differs from its pinned local resource")
