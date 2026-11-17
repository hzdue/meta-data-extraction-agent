# Codex-assisted pseudo-label semantic review

## Scope

- Reviewed the evidence-screen output for all 28 stratified papers.
- The sample contains four papers from each subject group and eight validation papers.
- Exact normalized matches: 289/326 checked strings.
- Directly inspected the lowest-match documents and the label-quality flag groups.
- Visually inspected the worst parser-coverage pages in `1903.05884`, `2207.14409` and `1910.09611`.

## Findings

- Exact-match misses were attributable to PDF typography, formulas, abbreviations, paraphrased technique labels or line-breaking; they remain visible in the review JSONL.
- Repeated sample names in `2208.02456`, `2603.09635` and `2604.26560` refer to distinct physical forms such as single crystal, powder or polycrystalline material.
- Repeated dataset names in `1809.02567`, `1909.06555`, `2002.08152`, `2101.08026`, `2108.11506` and `2602.22156` refer to distinct campaigns, samples, facilities or acquisition conditions; the quality flags are retained.
- The 27 samples in `2212.01894` are explicitly named variants rather than an obvious duplicated inventory.
- `2606.10715` has no named facility but contains laboratory-style datasets; an empty facility list is allowed by the schema.
- The visually reviewed low-coverage pages contain ordinary scientific text, tables, figures or sparse supplementary pages. No additional parser exclusion was warranted.

## Decision

No additional records were excluded or edited. All quality flags remain attached to their training records. This review supports a provisional pilot, not a claim that the pseudo-labels are scientifically equivalent to human annotations. An independent domain-expert review remains recommended before publication-quality use.
