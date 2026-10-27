# Step 2 PDF extraction review

Reviewed on 2026-10-27. All 40 complete PDFs were processed by the unchanged
repository `PdfParser`. No page ranges, truncation, manual text repairs, or gold
answers were used to alter extraction. Source/text hashes and parser settings are
recorded in `preparation_manifest.json`.

## Scope and findings

PyMuPDF-rendered previews of 15 pages across 10 papers were inspected alongside
their Markdown. This is a targeted spot check, not a full transcription audit.
Poppler was unavailable; previews used the installed PyMuPDF renderer.

| Paper | PDF pages inspected (1-based) | Finding |
|---|---|---|
| 1806.08116 | 20 | Low word-presence flag comes from mathematical/subscript labels being split by Markdown markup. Table 5's caption, compositions, rows and numerical values are present. |
| 1812.03956 | 1 | Font warning; title, authors and affiliations appear in the extracted front matter. |
| 1901.08819 | 1 | Font warning; front matter is present, but affiliation line boundaries are imperfect. |
| 2006.12074 | 1, 2 | PDF syntax warnings; both pages render and the corresponding experimental text, captions and references are present. Figures themselves are not model inputs. |
| 2304.08002 | 1, 11 | Compressed-stream warnings; front matter and end matter remain readable. Superscripts/hyphenation are imperfect. |
| 2309.13729 | 1 | Font warnings and a genuine title reading-order error: title words are reordered in Markdown. Kept unchanged as shared baseline-parser noise, not repaired using gold. |
| 2209.06540 | 18, 20 | Figure-heavy pages explain low word-presence scores. Figure labels/captions are present, but the Figure 5 caption has repeated text fragments. This is a documented extraction defect, not omitted pages. |
| 2407.10282 | 1 | Front matter is readable. Color-space/compressed-stream warnings elsewhere in the document remain logged; this spot check does not establish fidelity of all figures. |
| 2605.14792 | 1 | Front matter is readable. Font/ActualText warnings elsewhere remain logged; complete text fidelity is not guaranteed. |
| 2503.18475 | 1, 33, 65 | Front matter, middle reference section and final supplementary Figure S9 caption are present. The complete 65-page input was retained. |

## Interpretation

- The word-presence diagnostic compares each page's native text vocabulary with
  the whole Markdown document. It is only a warning heuristic: it neither verifies
  reading order nor proves that every occurrence/value was recovered.
- Graphical content is not supplied as images. The existing parser may preserve
  captions and embedded picture text, but not visual information contained only
  in plots or diagrams. This is a limitation of the shared text-only experiment.
- Retain these known parser imperfections consistently across M0/M1/M2. Results
  measure extraction from this fixed parsed representation, not an ideal PDF
  transcription. Changing the parser or repairing individual papers would require
  new text hashes, dataset regeneration and a new length audit.
- No annotations, frozen split membership, benchmark prompts or parser code were
  modified. Test annotations were not consulted for this review or target creation.

Length checks passed for all prepared examples and inference prompts. This review
does not certify GPU memory feasibility or scientific completeness of the gold.
