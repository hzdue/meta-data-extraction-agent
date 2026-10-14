# Pinned NeXus declarations for M05

The selected resource is **NeXus definitions v2026.01**, pinned to official
commit `0ad0f784950eefea5675e725d2ecb10a5f83e8ad`. The
[official tag reference](https://api.github.com/repos/nexusformat/definitions/git/ref/tags/v2026.01)
resolved to that commit on 2026-10-07. This is a selected version, not a claim
that the latest available definitions were selected.

Only three NXDL files were retrieved. They supply sample metadata, incident
beam terminology, and one monochromatic powder-diffraction application context.
The loader does not search the repository or fetch imports at runtime.

## Local pins and attribution

Configuration: `configs/agent/nexus_resource.json`.
Unmodified source files are retained under the ignored directory
`runs/agent/m05-20261007-nexus-v1/`. All bytes came from the official
GitHub public file API at the pinned commit. Base64 decoding preserved bytes;
each file's Git blob SHA-1 was recomputed and matched the API response.

| File / official pinned source | Bytes | SHA-256 | Git blob SHA-1 |
| --- | ---: | --- | --- |
| [NXsample.nxdl.xml](https://github.com/nexusformat/definitions/blob/0ad0f784950eefea5675e725d2ecb10a5f83e8ad/base_classes/NXsample.nxdl.xml) | 14927 | `3de15da384c1611cfbf5f230beb127ccfd7b5612026e46e74fc61181cbbe747a` | `d7721a53faf91bd4e59ec1cd283839730b42a092` |
| [NXbeam.nxdl.xml](https://github.com/nexusformat/definitions/blob/0ad0f784950eefea5675e725d2ecb10a5f83e8ad/base_classes/NXbeam.nxdl.xml) | 19122 | `f180db9ea264fb9534f8022db2444f25f85e1e09d00be884cc9eff2273983682` | `e14afec573c23ab678618ab3272f598a61595072` |
| [NXmonopd.nxdl.xml](https://github.com/nexusformat/definitions/blob/0ad0f784950eefea5675e725d2ecb10a5f83e8ad/applications/NXmonopd.nxdl.xml) | 5487 | `9abfdea918c72c129912e7069dbe4fa9e72159f5a23fefe66f203b4d8e406616` | `8419cd952f32d7056fe151984c3e0687e683dd51` |
| [LGPL.txt](https://github.com/nexusformat/definitions/blob/0ad0f784950eefea5675e725d2ecb10a5f83e8ad/LGPL.txt) | 7651 | `da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768` | `65c5ca88a67c30becee01c5a8816d964b03862f9` |
| [COPYING](https://github.com/nexusformat/definitions/blob/0ad0f784950eefea5675e725d2ecb10a5f83e8ad/COPYING) | 13 | `3e508fd9b5d3e70dbc974a54d3e87103a08fcd4f1f9adf6b7f13dfa6400ee6f0` | `8deff7dac57530f5fb448d155791e7bafe5c42ac` |

The original XML notices attribute copyright to the NeXus International Advisory
Committee (NIAC), 2008–2026, and specify GNU LGPL version 3 or later. They remain
intact. The same-commit `LGPL.txt` is also pinned and verified by the loader;
`COPYING` points to it and is retained beside it. This resource attribution
does not resolve the inherited repository's empty root `LICENSE`, and the
resource files are not added to commits or redistributed here.

A guessed `applications/NXxpd.nxdl.xml` path returned 404 during preparation.
The official application directory listing then identified `NXmonopd`;
no nonexistent definition was fabricated. An initial combined file-write tool
call exceeded the automatic review's 64000-byte input bound; splitting the same
authorized data into individual writes succeeded. No model inference was used.

## Projected context and limits

Parser version: `m05-nxdl-v1`.
Namespace: `http://definition.nexusformat.org/nxdl/3.1`.
Snapshot identity:

```text
nxdl:8a779d11868e2f94d45b0cc6d24285c0cf157c1d035ee1bf58b1bcee7e54bc3b
```

The snapshot contains **85 explicit fields**: NXsample 39, NXbeam 32, and
NXmonopd 14. Each field includes its declaring class and category, exact XML
context, class and field documentation, stated type/units/occurrence attributes,
group context, dimension XML, original source SHA-256, commit and release, and
a content-derived identity. Root/class attributes are retained. Unspecified
types, units, and numeric occurrence bounds remain absent.

For example, `NXsample/temperature` declares `NX_FLOAT` and
`NX_TEMPERATURE`; its documentation discusses a sample temperature that may
be scanned. This does not establish that a paper reported any temperature,
nor that a preparation temperature can fill an acquisition field.
`NXbeam/incident_wavelength` retains its incident-beam context.
The explicit detector-data declaration in NXmonopd is addressed as
`NXentry/NXinstrument/NXdetector/data` within that application definition;
this is **not a universal HDF5 instance path**. Its `NX_INT` representation
and dimensional constraints cannot supply missing experimental counts.
[Official NXDL overview](https://manual.nexusformat.org/nxdl.html),
[NXsample documentation](https://manual.nexusformat.org/classes/base_classes/NXsample.html).

Base definitions list possibilities; application definitions describe a
particular application. The loader retains that distinction but does not
apply XSD/category defaults, assess complete application compliance, or
generate HDF5. In the selected files, NXsample extends NXcomponent;
NXbeam and NXmonopd extend NXobject. These inheritances, group-type expansions, links,
and includes are explicitly unresolved even if a referenced class happens
to be selected. Unrelated group limits stay class-level; each field reports
its own enclosing unresolved dependencies. This makes directly declared
content useful for proposals without pretending a complete resolved schema.

Dimension expressions are preserved, not evaluated. Field attributes,
class/group attribute declarations, choice alternatives, symbol semantics,
and other unprojected constraints are explicitly flagged where relevant.
Enumeration values are retained; additional enumeration constraints are not
silently flattened into a complete closed constraint. Declared
`recommended=true` is treated as an optional recommendation, as documented
by [the official NXDL element reference](https://manual.nexusformat.org/nxdl_desc.html);
`recommended=false` alone does not establish a minimum occurrence.
Conflicting explicit required/optional declarations fail.

Files are limited to 1 MB each and at most 12 selected NXDL files. XML depth
and element-count limits, an aggregate 4 MB input/projection accounting bound,
strict UTF-8, and rejection of DTD/entity declarations bound parsing.
No XSD imports, XML stylesheets, document instructions, network calls,
or arbitrary generated code execute. Test XML uses explicit
`urn:synthetic:` namespaces and is not presented as an authentic standard.

## Reproduction and checks

Prepare these exact official files in a **fresh ignored agent run directory**,
preserving bytes; update only a deliberate resource manifest if its local
path changes. Verify the listed SHA-256 values and retain attribution/license
files. The runtime config never downloads missing resources. Missing or
changed files fail explicitly, including license-file changes.

The focused resource suite has **68 offline tests**. The joint resource,
registry, and runner suite passed **192 tests** after the projection hardening.
The authentic snapshot was loaded and `verify_resources(snapshot, root)`
passed separately without claiming an inspection run. Root integration
commands and final full-suite results are recorded in `COMMANDS.md`.

These are engineering checks. They do not demonstrate extraction improvement,
independent expert review, or completed E05 evaluation.
