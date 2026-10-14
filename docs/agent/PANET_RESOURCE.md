# PaNET resource used by M04

Retrieved 2026-10-07 from the official [PaNET repository](https://github.com/pan-ontologies/PaNET).
This is a pinned historical resource, not a claim to use the latest ontology.

| Property | Recorded value |
|---|---|
| Selected release | `v1.1.0` |
| Commit | `6d23ae3ea9ab62f4e8be97cc38c2ebdf0367c142` |
| Upstream file | [`releases/latest-release/PaNET.owl` at that commit](https://github.com/pan-ontologies/PaNET/blob/6d23ae3ea9ab62f4e8be97cc38c2ebdf0367c142/releases/latest-release/PaNET.owl) |
| Local file | `runs/agent/m04-20261007-panet-v1/PaNET.owl` |
| Raw size | 213133 bytes |
| SHA-256 | `ef4dbea93c9e16883114fb1fc6132fc79797382351d8796d0daf827a1b413ddb` |
| Verified Git blob SHA-1 | `76cb59a3a9c1f389736f2945922aaafeb963b7d9` |
| Term namespace | `http://purl.org/pan-science/PaNET/` |
| Named concepts loaded | 374 |
| Config | `configs/agent/panet_resource.json` |

The downloaded bytes were reconstructed through the available GitHub read tool;
their Git blob hash matches the official tree entry. Runtime verifies SHA-256
against the explicit configuration. It performs no fetch, repository indexing or
automatic import resolution. The configuration is a trusted resource declaration,
not an independent cryptographic certificate of upstream ownership.

The [v1.3.0 release](https://github.com/pan-ontologies/PaNET/releases/tag/v1.3.0)
exists. Its generated asset was not obtainable through the available download
route, so this module uses the accessible immutable v1.1.0 file. A later upgrade
requires a new resource pin, changed cache identities and explicit rechecking.
The selected file's own `owl:versionInfo` includes `1.0` and a first-version note;
the loader preserves these annotations separately from the release label.

## Attribution and resource terms

Credit: PaNET contributors, the Photon and Neutron Experimental Techniques
ontology, as attributed in the unmodified upstream RDF/XML. Creator/contributor
annotations and notices are retained in the local copy. The ontology metadata
states [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
That license calls for attribution, a license link and indication of changes when
sharing adaptations. This module leaves the downloaded OWL bytes unchanged and
creates a separately identified parser projection.

The available [upstream license text](https://github.com/pan-ontologies/PaNET/blob/7f88345b40242f77dc91511ea8abadf518ae3ff0/LICENSE)
was saved as `LICENSE.cc-by-4.0.txt` beside the OWL file: 18658 bytes,
SHA-256 `f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f`.
Its commit differs from the ontology commit because that older tag has no LICENSE
file; the ontology's own license annotation establishes the recorded resource
terms. Raw third-party resources remain in ignored local storage. The project's
empty `LICENSE` and benchmark redistribution questions remain unresolved; this
resource record does not assign a license to the repository or its data.

## Projection and scope

`panet_resources.py` loads explicit named classes, English/unqualified labels,
available alternative/synonym annotations, definitions, direct named parents and
deprecation/replacement metadata. A SKOS alternative label stays `alternative`,
not an asserted exact synonym. DTD/entity declarations are rejected. Imports,
restrictions, intersections and equivalent-class axioms are not evaluated; foreign
parents can be recorded as links without loading their definitions.

The demonstration selects two explicit families:

- Powder diffraction: `http://purl.org/pan-science/PaNET/PaNET01030`.
- Small angle scattering: `http://purl.org/pan-science/PaNET/PaNET01124`.

For example, x-ray powder diffraction `PaNET01101` has the alternative label
`XPD`. Both facts come from the pinned resource. They do not establish that any
publication used that technique. The projected named-parent graph is incomplete
as an OWL classification; the module deliberately does not infer absent links.
Every mapping remains an unapproved sidecar proposal with publication evidence
kept separate from ontology definitions.
