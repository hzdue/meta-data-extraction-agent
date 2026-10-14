"""Bounded, pinned NXDL declarations for guidance; no inheritance or schema engine."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from pydantic import Field, model_validator

from .contracts import digest
from .evidence import FrozenModel, sha256
from .paths import PathPolicyError, input_file

PARSER_VERSION = "m05-nxdl-v1"
NXDL_NAMESPACE = "http://definition.nexusformat.org/nxdl/3.1"
MAX_RESOURCE_BYTES = 1_000_000
MAX_PROJECTION_BYTES = 4_000_000
MAX_ELEMENTS = 10000
MAX_DEPTH = 32
Category = Literal["base", "application", "contributed"]


class NexusResourceError(ValueError):
    """An explicitly selected NXDL resource or definition cannot be used."""


class NXDLFileSpec(FrozenModel):
    path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_url: str = Field(min_length=1, max_length=1500)


class NXDLResourceSpec(FrozenModel):
    kind: Literal["authentic", "synthetic"]
    commit: str = Field(min_length=1, max_length=100)
    release: str = Field(min_length=1, max_length=100)
    files: tuple[NXDLFileSpec, ...] = Field(min_length=1, max_length=12)
    license_name: str = Field(min_length=1, max_length=200)
    license_url: str = Field(min_length=1, max_length=1500)
    license_file: NXDLFileSpec | None = None

    @model_validator(mode="after")
    def _provenance(self) -> NXDLResourceSpec:
        paths = [f.path.replace("\\", "/") for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("Resource files must be unique")
        if self.kind == "authentic":
            if not re.fullmatch(r"[a-f0-9]{40}", self.commit):
                raise ValueError("Authentic NXDL requires an exact repository commit")
            if self.license_name != "LGPL-3.0-or-later" or self.license_file is None:
                raise ValueError("Authentic NXDL requires the recorded pinned license")
            expected = f"https://raw.githubusercontent.com/nexusformat/definitions/{self.commit}/LGPL.txt"
            if self.license_url != expected or self.license_file.source_url != expected:
                raise ValueError("License URL must pin the same official commit")
            for item in (*self.files, self.license_file):
                url = urlsplit(item.source_url)
                prefix = f"/nexusformat/definitions/{self.commit}/"
                if (url.scheme != "https" or url.netloc != "raw.githubusercontent.com"
                        or not url.path.startswith(prefix) or url.query or url.fragment):
                    raise ValueError("Authentic URL must pin an official definitions file")
                suffix = url.path[len(prefix):]
                if item is not self.license_file and not re.fullmatch(
                        r"(?:base_classes|applications|contributed)/NX[A-Za-z0-9_]+\.nxdl\.xml", suffix):
                    raise ValueError("Only explicitly selected NXDL definitions are allowed")
                if not item.path.replace("\\", "/").startswith("runs/agent/"):
                    raise ValueError("Authentic resource files belong in agent runs")
        else:
            for item in (*self.files, *((self.license_file,) if self.license_file else ())):
                if (not item.path.replace("\\", "/").startswith("tests/agent/fixtures/")
                        or not item.source_url.startswith("urn:synthetic:")):
                    raise ValueError("Synthetic resources require fixture paths and synthetic provenance")
            if not self.license_url.startswith("urn:synthetic:"):
                raise ValueError("Synthetic license provenance must be explicit")
        return self


class NXDLGroup(FrozenModel):
    name: str | None = None
    nx_class: str
    context_path: str
    documentation: str | None = None
    min_occurs: str | None = None
    max_occurs: str | None = None
    optional: str | None = None
    recommended: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()


class NXDLField(FrozenModel):
    definition_id: str = Field(pattern=r"^nxdl-field:[a-f0-9]{64}$")
    class_name: str
    category: Category
    field_path: str
    field_name: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_url: str
    resource_commit: str
    resource_release: str
    namespace: str
    schema_version: str
    class_documentation: str | None = None
    class_extends: str | None = None
    documentation: str | None = None
    nx_type: str | None = None
    units: str | None = None
    min_occurs: str | None = None
    max_occurs: str | None = None
    optional: str | None = None
    recommended: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    group_context: tuple[NXDLGroup, ...] = ()
    dimensions_xml: str | None = None
    enumeration: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    reference_kind: Literal["nxdl_declaration_not_universal_hdf5_path"] = "nxdl_declaration_not_universal_hdf5_path"

    @model_validator(mode="after")
    def _identity(self) -> NXDLField:
        if self.definition_id != "nxdl-field:" + digest(self.model_dump(mode="json", exclude={"definition_id"})):
            raise ValueError("Field identity differs from its definition content")
        return self


class NXDLClass(FrozenModel):
    class_name: str
    category: Category
    namespace: str
    schema_version: str
    extends: str | None = None
    documentation: str | None = None
    source_sha256: str
    source_url: str
    attributes: tuple[tuple[str, str], ...] = ()
    fields: tuple[NXDLField, ...] = ()
    unresolved: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _integrity(self) -> NXDLClass:
        paths = [field.field_path for field in self.fields]
        if paths != sorted(set(paths)):
            raise ValueError("Field contexts must be unique and sorted")
        if any(f.class_name != self.class_name or f.category != self.category
               or f.source_sha256 != self.source_sha256 or f.source_url != self.source_url
               or f.namespace != self.namespace or f.schema_version != self.schema_version
               or f.class_documentation != self.documentation or f.class_extends != self.extends
               for f in self.fields):
            raise ValueError("Field context differs from its declaring class")
        return self


class NXDLSnapshot(FrozenModel):
    resource: NXDLResourceSpec
    snapshot_id: str = Field(pattern=r"^nxdl:[a-f0-9]{64}$")
    classes: tuple[NXDLClass, ...] = Field(min_length=1, max_length=12)
    limitations: tuple[str, ...]
    parser_version: Literal["m05-nxdl-v1"] = PARSER_VERSION

    @model_validator(mode="after")
    def _integrity(self) -> NXDLSnapshot:
        names = [c.class_name for c in self.classes]
        if names != sorted(set(names)):
            raise ValueError("Definition classes must be unique and sorted")
        pins = {(f.sha256, f.source_url) for f in self.resource.files}
        if len(self.classes) != len(pins) or any((c.source_sha256, c.source_url) not in pins for c in self.classes):
            raise ValueError("Every class must correspond to exactly one selected resource")
        for cls in self.classes:
            if any(f.resource_commit != self.resource.commit or f.resource_release != self.resource.release for f in cls.fields):
                raise ValueError("Field resource version differs from snapshot")
        if self.snapshot_id != "nxdl:" + digest(self.model_dump(mode="json", exclude={"snapshot_id"})):
            raise ValueError("NXDL snapshot identity differs from its content")
        return self


def _doc(element: ET.Element, namespace: str) -> str | None:
    docs = element.findall("{" + namespace + "}doc")
    # Keep source wording and internal precision; trim only outer XML indentation.
    return "\n\n".join("".join(node.itertext()).strip() for node in docs).strip() or None


def _occurs(element: ET.Element) -> dict[str, str | None]:
    minimum = element.get("minOccurs")
    maximum = element.get("maxOccurs")
    if minimum is not None and not re.fullmatch(r"[0-9]+", minimum):
        raise NexusResourceError("Invalid explicit NXDL minimum occurrence")
    if maximum is not None and not re.fullmatch(r"(?:[0-9]+|unbounded)", maximum):
        raise NexusResourceError("Invalid explicit NXDL maximum occurrence")
    if minimum is not None and maximum not in {None, "unbounded"} and int(minimum) > int(maximum):
        raise NexusResourceError("Inconsistent explicit NXDL occurrence limits")
    optional = element.get("optional")
    recommended = element.get("recommended")
    for value in (optional, recommended):
        if value not in {None, "true", "false", "1", "0"}:
            raise NexusResourceError("Invalid explicit NXDL optional/recommended flag")
    if (optional in {"true", "1"} or recommended in {"true", "1"}) and minimum is not None and int(minimum) > 0:
        raise NexusResourceError("Conflicting explicit NXDL occurrence flags")
    if optional in {"false", "0"} and minimum == "0":
        raise NexusResourceError("Conflicting explicit NXDL occurrence flags")
    if recommended in {"true", "1"} and optional in {"false", "0"}:
        raise NexusResourceError("Conflicting explicit NXDL occurrence flags")
    return {"min_occurs": minimum, "max_occurs": maximum, "optional": optional, "recommended": recommended}


def _name(value: str | None, *, nx_class: bool = False) -> str:
    if value is None or not re.fullmatch(r"NX[A-Za-z0-9_]+" if nx_class else r"[A-Za-z_][A-Za-z0-9_.]*", value):
        raise NexusResourceError("Unsupported or absent NXDL declaration name")
    return value


def _parse(raw: bytes, file: NXDLFileSpec, spec: NXDLResourceSpec, budget: list[int]) -> NXDLClass:
    budget[0] += len(raw)
    if budget[0] > MAX_PROJECTION_BYTES:
        raise NexusResourceError("NXDL projected content exceeds the aggregate size bound")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise NexusResourceError("NXDL input must be UTF-8 XML") from exc
    if "\x00" in text or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.I):
        raise NexusResourceError("NXDL DTD and entity declarations are forbidden")
    declaration = re.search(r"<\?xml\s[^?]*encoding\s*=\s*['\"]([^'\"]+)", text, re.I)
    if declaration and declaration.group(1).lower().replace("_", "-") not in {"utf-8", "utf8"}:
        raise NexusResourceError("NXDL input declares an unsupported encoding")
    try:
        document = ET.fromstring(text)
    except ET.ParseError as exc:
        raise NexusResourceError("Malformed NXDL XML") from exc
    match = re.fullmatch(r"\{([^}]+)\}definition", document.tag)
    if not match:
        raise NexusResourceError("Expected a namespaced NXDL definition root")
    namespace = match.group(1)
    if ((spec.kind == "authentic" and namespace != NXDL_NAMESPACE)
            or (spec.kind == "synthetic" and not namespace.startswith("urn:synthetic:"))):
        raise NexusResourceError("NXDL namespace does not match the resource kind")
    stack = [(document, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > MAX_DEPTH or count > MAX_ELEMENTS:
            raise NexusResourceError("NXDL nesting or element count exceeds the projection bound")
        stack.extend((child, depth + 1) for child in node)
    name = _name(document.get("name"), nx_class=True)
    category = document.get("category")
    if category not in {"base", "application", "contributed"}:
        raise NexusResourceError("Expected an explicit NXDL category")
    if spec.kind == "authentic":
        expected_folder = {"base": "base_classes", "application": "applications", "contributed": "contributed"}[category]
        if not file.source_url.endswith(f"/{expected_folder}/{name}.nxdl.xml"):
            raise NexusResourceError("NXDL class/category differs from the pinned file URL")
    extends = document.get("extends")
    if extends:
        _name(extends, nx_class=True)
    schema_version = namespace.rsplit("/", 1)[-1] if spec.kind == "authentic" else "synthetic"
    tag = lambda local: "{" + namespace + "}" + local
    root_doc = _doc(document, namespace)
    class_unresolved = set()
    if extends:
        class_unresolved.add(f"Inheritance from {extends} is unresolved; declarations are not expanded.")
    # Never fetch XInclude, other XML imports, or arbitrary referenced resources.
    has_includes = any(n.tag.rsplit("}", 1)[-1] in {"include", "import"} for n in document.iter())
    for node in document.iter():
        local = node.tag.rsplit("}", 1)[-1]
        if local in {"include", "import"}:
            class_unresolved.add("Includes/imports are unresolved and were not fetched.")
        if local == "symbols":
            class_unresolved.add("Symbol definitions are retained only in the source; expressions are not evaluated.")
        if local == "choice":
            class_unresolved.add("Choice alternatives are not projected.")
    fields = []

    def walk(parent: ET.Element, context: tuple[NXDLGroup, ...], inherited_attributes: bool = False) -> None:
        seen = set()
        parent_unresolved = inherited_attributes or parent.find(tag("attribute")) is not None
        if parent_unresolved:
            class_unresolved.add("Class/group attribute declarations are not projected.")
        for node in parent:
            if node.tag == tag("group"):
                nx_class = _name(node.get("type"), nx_class=True)
                group_name = node.get("name")
                if group_name is not None:
                    _name(group_name)
                segment = f"{group_name}:{nx_class}" if group_name else nx_class
                if ("group", segment) in seen:
                    raise NexusResourceError("Ambiguous duplicate NXDL group context")
                seen.add(("group", segment))
                path = "/".join([*(g.context_path.rsplit("/", 1)[-1] for g in context), segment])
                group = NXDLGroup(name=group_name, nx_class=nx_class, context_path=path,
                                  documentation=_doc(node, namespace),
                                  attributes=tuple(sorted(node.attrib.items())), **_occurs(node))
                class_unresolved.add(f"Group reference {path} ({nx_class}) is unresolved; base declarations are not expanded.")
                walk(node, (*context, group), parent_unresolved)
            elif node.tag == tag("field"):
                field_name = _name(node.get("name"))
                if ("field", field_name) in seen:
                    raise NexusResourceError("Duplicate NXDL field declaration")
                seen.add(("field", field_name))
                field_path = "/".join([*(g.context_path.rsplit("/", 1)[-1] for g in context), field_name])
                dimensions = node.findall(tag("dimensions"))
                if len(dimensions) > 1:
                    raise NexusResourceError("Ambiguous NXDL dimensions declarations")
                unresolved = set()
                if extends:
                    unresolved.add(f"Inheritance from {extends} is unresolved; declarations are not expanded.")
                if has_includes:
                    unresolved.add("Includes/imports are unresolved and were not fetched.")
                unresolved.update(f"Group reference {g.context_path} ({g.nx_class}) is unresolved; base declarations are not expanded."
                                  for g in context)
                if parent_unresolved or document.find(tag("attribute")) is not None:
                    unresolved.add("Enclosing class/group attribute declarations are not projected.")
                if any(n.tag == tag("attribute") for n in node):
                    unresolved.add("Nested field attribute definitions are not projected.")
                if any(n.tag not in {tag(x) for x in ("doc", "dimensions", "enumeration", "attribute")} for n in node):
                    unresolved.add("Unsupported field child constraints are not projected.")
                if any(enum.attrib or any(n.tag != tag("item") or set(n.attrib) != {"value"} or len(n) for n in enum)
                       for enum in node.findall(tag("enumeration"))):
                    unresolved.add("Enumeration annotations/constraints beyond item values are not projected.")
                if dimensions:
                    unresolved.add("Dimension expressions are retained as XML, not evaluated.")
                if node.get("type") is None:
                    unresolved.add("Type is not explicit; NXDL schema defaults/inheritance are not applied.")
                if node.get("minOccurs") is None and node.get("optional") is None and node.get("recommended") not in {"true", "1"}:
                    unresolved.add("Minimum occurrence is not explicit; category/schema defaults are not applied.")
                values = tuple(n.get("value") for enum in node.findall(tag("enumeration"))
                               for n in enum.findall(tag("item")) if n.get("value") is not None)
                payload = dict(class_name=name, category=category, field_path=field_path,
                               field_name=field_name, source_sha256=file.sha256, source_url=file.source_url,
                               resource_commit=spec.commit, resource_release=spec.release,
                               namespace=namespace, schema_version=schema_version,
                               class_documentation=root_doc, class_extends=extends,
                               documentation=_doc(node, namespace), nx_type=node.get("type"),
                               units=node.get("units"), attributes=tuple(sorted(node.attrib.items())),
                               group_context=context,
                               dimensions_xml=ET.tostring(dimensions[0], encoding="unicode").strip() if dimensions else None,
                               enumeration=values, unresolved=tuple(sorted(unresolved)),
                               reference_kind="nxdl_declaration_not_universal_hdf5_path", **_occurs(node))
                plain = {**payload, "group_context": [g.model_dump(mode="json") for g in context]}
                budget[0] += len(json.dumps(plain, ensure_ascii=False).encode("utf-8"))
                if budget[0] > MAX_PROJECTION_BYTES:
                    raise NexusResourceError("NXDL projected content exceeds the aggregate size bound")
                fields.append(NXDLField(**payload, definition_id="nxdl-field:" + digest(plain)))
            elif node.tag == tag("link"):
                class_unresolved.add("Link targets are unresolved and are not projected as field declarations.")
            elif node.tag not in {tag(x) for x in ("doc", "symbols", "attribute", "choice")}:
                class_unresolved.add("Unsupported NXDL/XML constructs are not projected.")
    walk(document, ())
    return NXDLClass(class_name=name, category=category, namespace=namespace, schema_version=schema_version,
                     extends=extends, documentation=root_doc, source_sha256=file.sha256, source_url=file.source_url,
                     attributes=tuple(sorted(document.attrib.items())),
                     fields=tuple(sorted(fields, key=lambda f: f.field_path)), unresolved=tuple(sorted(class_unresolved)))


def _read(root: Path | str, file: NXDLFileSpec, kind: str) -> bytes:
    try:
        path = input_file(root, file.path, allowed_roots=("runs/agent",) if kind == "authentic" else ("tests/agent/fixtures",))
        if path.stat().st_size > MAX_RESOURCE_BYTES:
            raise NexusResourceError("NXDL resource exceeds the bounded file size")
        raw = path.read_bytes()
    except (OSError, PathPolicyError) as exc:
        raise NexusResourceError("Pinned NXDL resource is unavailable at its approved path") from exc
    if len(raw) > MAX_RESOURCE_BYTES:
        raise NexusResourceError("NXDL resource exceeds the bounded file size")
    if sha256(raw) != file.sha256:
        raise NexusResourceError("NXDL resource bytes differ from the pinned SHA-256")
    return raw


def load_nexus(root: Path | str, spec: NXDLResourceSpec) -> NXDLSnapshot:
    """Read only the explicit allowlist; never search directories or fetch a URL."""
    spec = NXDLResourceSpec.model_validate(spec.model_dump())
    if spec.license_file:
        _read(root, spec.license_file, spec.kind)
    budget = [0]
    classes = tuple(sorted((_parse(_read(root, file, spec.kind), file, spec, budget) for file in spec.files),
                           key=lambda c: c.class_name))
    payload = {"resource": spec.model_dump(mode="json"), "classes": [c.model_dump(mode="json") for c in classes],
               "limitations": [
                   "Only explicit declarations are projected; includes, inheritance, group references and links are not expanded.",
                   "Unstated types and occurrences remain unknown; NXDL schema/category defaults are not applied.",
                   "Declaration references are context addresses, not universal HDF5 paths.",
                   "No NXDL XSD validation or application-compliance assessment is performed.",
               ], "parser_version": PARSER_VERSION}
    try:
        return NXDLSnapshot.model_validate({**payload, "snapshot_id": "nxdl:" + digest(payload)})
    except ValueError as exc:
        raise NexusResourceError("Inconsistent or duplicate NXDL definitions") from exc


def retrieve_definition(snapshot: NXDLSnapshot, class_name: str, field_path: str) -> NXDLField:
    """Return an exact declared context, without fuzzy aliases or inherited expansion."""
    snapshot = NXDLSnapshot.model_validate(snapshot.model_dump())
    _name(class_name, nx_class=True)
    for cls in snapshot.classes:
        if cls.class_name == class_name:
            for field in cls.fields:
                if field.field_path == field_path:
                    return field
    raise NexusResourceError("Requested NXDL field context is unavailable in the pinned declarations")


def verify_resources(snapshot: NXDLSnapshot, root: Path | str) -> None:
    """Reject changed pins, bytes, declarations or parser projections."""
    snapshot = NXDLSnapshot.model_validate(snapshot.model_dump())
    if load_nexus(root, snapshot.resource) != snapshot:
        raise NexusResourceError("NXDL snapshot differs from pinned local resources")
