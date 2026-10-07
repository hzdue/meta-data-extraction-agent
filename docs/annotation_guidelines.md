# Annotation Guidelines

Use the field definitions in [the generic schema](../schemas/generic.json) and
[the photon and neutron schema](../schemas/photon_and_neutron.json), both
version 1.1.0. The sections below present their field rules and constraints.

## Annotation workflow

1. Annotate independently from the assigned PDF. Use the PDF as the sole source 
   of scientific and bibliographic values. Filenames, corpus-index entries, 
   extracted abstracts, and external knowledge do not support a value absent from 
   the PDF.
2. Save the record as `data/annotations/<annotator_id>/<arxiv_id>.json` using the
   schema fields and types. Follow each field's rules for missing values, empty
   lists, and allowed values.


## Generic Schema Fields

### `bibliographic.title`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Main title of the scientific document itself.

**Instructions:**
- Copy the document title verbatim as printed on the first page or running header. Preserve casing and punctuation.

### `bibliographic.subtitle`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Subtitle of the document, if explicitly given.

### `bibliographic.affiliations`

**Type:** `list[affiliation]`
**Cardinality:** min `0`, max `100`

**Description:** Institutional affiliations associated with document authors.

**Instructions:**
- One entry per distinct affiliation, in order of first appearance. Do not repeat affiliations.

### `bibliographic.affiliations[].id`

**Type:** `string`

**Description:** Stable local affiliation identifier.

**Instructions:**
- Assign IDs in sequence: aff_1, aff_2, aff_3, etc.

### `bibliographic.affiliations[].label`

**Type:** `string|null`

**Description:** Local marker used in the document, such as a superscript number, letter, or symbol.

**Instructions:**
- Copy the printed superscript or symbol exactly as printed, e.g. '1', 'a', '†', '‡'. Use null if no explicit label is given.

### `bibliographic.affiliations[].text`

**Type:** `string`

**Description:** Affiliation text as given in the document.

**Instructions:**
- Copy the full affiliation address verbatim, including department, institution, postal code, city, country.

### `bibliographic.authors`

**Type:** `list[author]`
**Cardinality:** min `0`, max `100`

**Description:** Authors of the scientific document in document order.

**Instructions:**
- Extract document authors only.
- Preserve author order as it appears in the document.

### `bibliographic.authors[].id`

**Type:** `string`

**Description:** Stable local author identifier.

**Instructions:**
- Assign IDs in sequence: author_1, author_2, author_3, etc.

### `bibliographic.authors[].full_name`

**Type:** `string`

**Description:** Full author name as presented in the document.

**Instructions:**
- Copy as printed. Use initials if the document prints initials; full names if it prints full names. Do not expand initials.

### `bibliographic.authors[].orcid`

**Type:** `string|null`

**Description:** ORCID identifier if explicitly given.

**Instructions:**
- Extract only ORCID identifiers explicitly associated with the author.
- Return ORCID as the bare identifier without URL prefix, e.g. 0000-0002-1825-0097.

**Constraints:** `pattern`: `^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$`

### `bibliographic.authors[].email`

**Type:** `string|null`

**Description:** Author email if explicitly given.

### `bibliographic.authors[].is_corresponding`

**Type:** `boolean`

**Description:** Whether the author is explicitly marked as a corresponding author.

**Instructions:**
- Use true if the author is explicitly marked as a corresponding author (asterisk, footnote, 'Corresponding author' label). Use false otherwise.
- Never use null; this field is a required boolean.

### `bibliographic.authors[].affiliation_refs`

**Type:** `list[string]`

**Description:** References linking the author to affiliations.

**Instructions:**
- Use the exact id values assigned in the affiliations list (e.g. aff_1, aff_2).
- Use [] if the author has no affiliated institution.

### `bibliographic.abstract`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Main abstract of the scientific document.

**Instructions:**
- Copy the printed abstract verbatim.
- Do not include graphical abstracts, highlights, or keyword lists.

### `bibliographic.keywords`

**Type:** `list[string]`
**Cardinality:** min `0`, max `50`

**Description:** Explicit author-provided or publisher-provided keywords.

**Instructions:**
- Extract explicit keywords only.
- Do not infer keywords from the abstract or body text.

### `bibliographic.doi`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** DOI of the document itself, if explicitly given.

**Instructions:**
- Extract only the DOI that belongs to the document itself.
- Do not extract DOIs from cited references.

**Constraints:** `pattern`: `^10\.\d{4,9}/\S+$`

### `bibliographic.arxiv_id`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** arXiv identifier of the document, if explicitly given.

**Instructions:**
- Extract only the arXiv ID that belongs to the document itself.
- Use the unversioned bare arXiv ID, e.g. '1812.03956' not '1812.03956v1'.
- Do not extract arXiv IDs from cited references.

### `bibliographic.url`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Canonical URL of the document, if explicitly given.

**Instructions:**
- Extract only a URL that points to the document itself.
- Do not extract URLs from cited references.

### `bibliographic.publication_date`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Publication date of the document in the most precise form available.

**Instructions:**
- Extract the publication date for the document itself, not dates from cited references.
- If multiple dates are present, use this priority order:
  1. final journal publication date
  2. online publication date
  3. accepted date
  4. revised date
  5. received date
  6. manuscript-printed publication date (e.g. a `(Dated: ...)` line inside the PDF)
  7. arXiv submission date (use the version stamped on the PDF being annotated)
  8. copyright year
- When two dates of the same priority level appear, use the later one.
- Use the most precise date available in the selected source: YYYY-MM-DD, YYYY-MM, or YYYY.
- Do not combine dates from different sources. If only a year is available for the highest-priority source, use that year rather than a more precise lower-priority date.

**Worked example (arXiv preprint with both an internal date and the arXiv stamp):**
For arXiv:1912.11816v2, the PDF prints `(Dated: June 17, 2020)` in the manuscript body and carries the arXiv submission stamp `16 Jun 2020` in the margin. No journal version exists, so levels 1–5 do not apply. Level 6 (manuscript-printed date) outranks level 7 (arXiv submission stamp), giving `publication_date = "2020-06-17"`.

### `bibliographic.venue`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Primary scholarly venue or container of the document, such as a journal, conference, workshop, proceedings venue, or repository.

**Instructions:**
- For arXiv preprints, use arXiv.
- For published versions, use the journal/conference name as printed.
- Use null if no venue indicator is printed.

### `bibliographic.publisher`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Publisher or issuing organization of the document.

**Instructions:**
- Publisher name as printed in cite line, footer, or copyright notice (e.g. 'American Chemical Society', 'Elsevier'). Use null for arXiv-only papers.

### `bibliographic.language`

**Type:** `string|null`
**Cardinality:** min `0`, max `1`

**Description:** Primary language of the document content.

**Constraints:** `format`: `ISO 639-1 two-letter code, e.g. en, fr, de.`

### `bibliographic.funding`

**Type:** `list[funding_item]`
**Cardinality:** min `0`, max `50`

**Description:** Funding information explicitly stated in the document,
restricted to entries with both an identifiable funder and an explicit grant or
award identifier.

**Instructions:**
- Create a funding entry only when the document gives both an identifiable funder and an explicit grant or award identifier for it.
- Create one entry per (funder, grant_id) pair:
  - if a funder gives multiple grants, create one entry per grant_id;
  - if a single grant has multiple co-funders, create one entry per co-funder.
- Do not create an entry for a funder, institution, or acknowledgment named without an explicit grant or award identifier (e.g. general thanks to an institution, vendor, or a fellowship/position with no award number).
- The identifier must be a funding grant or award number; do not use identifiers that are not tied to financial support.

### `bibliographic.funding[].funder`

**Type:** `string`

**Description:** Funding organization name associated with the grant.

**Instructions:**
- Use the funding organization name as printed; use the full name if printed, and the acronym only when no full name is given.
- When both the full name and its acronym are printed, use the full name and omit the acronym.
- If no funding organization is named but a funding programme, scheme, or project is named as the source, use that programme, scheme, or project name as the funder.

### `bibliographic.funding[].grant_id`

**Type:** `string`

**Description:** Grant or award identifier explicitly given in the document.

**Instructions:**
- Single grant ID exactly as printed.
- If the printed identifier includes an associated project, programme, or acronym alongside the grant or award number, preserve the full printed identifier phrase as written, e.g. `834134 (WATUSO)` or `SIONA,C14/22/099`.

### `document_overview.section_headings`

**Type:** `list[string]`
**Cardinality:** min `0`, max `200`

**Description:** Top-level headings of the main scientific body in reading order.

**Instructions:**
- List the explicit top-level headings of the main scientific body in reading order, exactly as printed (preserve numbering, capitalisation, punctuation).
- Exclude front matter (Abstract) and back matter (Acknowledgements, Funding, Author Contributions, Conflict/Competing Interests, Data Availability, References/Bibliography).
- Exclude subsection headings, metadata blocks, and figure/table captions.
- Exclude supplementary materials, supporting information, and appendix sections (both the section heading itself and any headings within them).

### `document_overview.has_references`

**Type:** `boolean`

**Description:** Whether the document contains a references or bibliography section.

### `document_overview.has_figures`

**Type:** `boolean`

**Description:** Whether the document contains one or more figures.

### `document_overview.has_tables`

**Type:** `boolean`

**Description:** Whether the document contains one or more tables.

---

## Photon And Neutron Schema Fields

### `facilities`

**Type:** `list[facility]`
**Cardinality:** min `0`, max `20`

**Description:** Named facilities where experimental measurements reported in the document were performed.

**Instructions:**
- Create one entry per distinct named facility explicitly stated as a location where reported experimental measurements were performed or data were collected.
- A facility is 'named' when the document refers to it by a proper name, acronym, short name, or facility identifier that uniquely identifies it (e.g. 'European Synchrotron Radiation Facility', 'ESRF', 'PETRA III'). Generic references such as 'the synchrotron', 'a neutron source', or 'the X-ray facility' do not count as naming.
- Use the most complete facility name explicitly printed in the document. If the document prints both a full name and its acronym, use the full name only. If only an acronym, short name, or facility identifier is printed, use that form.
- If both a specific facility and an umbrella institution are given for the same measurement, record only the specific facility, e.g. PETRA III rather than DESY.
- If only an umbrella institution is given as the measurement location, record it.
- Do not infer facilities from external knowledge.
- Exclude cited facilities not used in this document.

### `facilities[].id`

**Type:** `string`

**Description:** Stable local facility identifier.

**Instructions:**
- Assign IDs sequentially in order of first appearance: facility_1, facility_2, facility_3, etc.

### `facilities[].name`

**Type:** `string`

**Description:** Facility name as stated in the document.

### `samples`

**Type:** `list[sample]`
**Cardinality:** min `0`, max `50`

**Description:** Samples studied through experimental measurements reported or reanalyzed in the document. A sample entry can represent a single specimen or a measured group when only the group is identified by the document.

**Instructions:**
- Follow this decision routine to determine sample entries.
- Step 1: Identify every material/specimen for which the document reports or reanalyzes experimental measurement data in its own study. Ignore materials mentioned only in background, references, or cited work.
- Step 2: For each measured material/specimen, ask 'does the document explicitly identify sample-defining variants?' Sample-defining variants include:
-   (a) different identified materials/specimens (e.g. different labels, names, compositions, material identities, or physical forms);
-   (b) conditions that produced distinct prepared specimens BEFORE measurement (different preparation routes, treatments, growth conditions, annealing conditions, or reduction conditions).
- Step 3: If no sample-defining variants are identified, create one sample entry for that material/specimen.
- Step 4: If sample-defining variants are identified, create one sample entry per explicitly identified variant.
- Step 5: When multiple sample-defining variant axes co-occur, create one sample entry per unique explicitly identified combination that the document actually prepared and measured. Do not create entries for possible combinations that are not explicitly identified as prepared and measured.
- Step 6: Do not create a duplicate entry for an alias, nickname, or descriptor (such as 'optimized sample', 'the best-performing film', 'the reference sample') if it refers to the same explicitly identified measured specimen.
- Step 7: Do not invent variants from ranges or formulas. A range expression like 'x = 0.05 to 0.20' identifies only the range as printed. Only variant values explicitly printed in the document text count as individual identifications.
- Step 8: Conditions varied DURING measurement of the same material/specimen (e.g. temperatures, pressures, magnetic fields, atmospheres, or other parameters changed while collecting data) are NOT sample-defining. Do not create separate sample entries.
- Step 9: For in-situ or operando measurements, do not create separate sample entries for phases, states, or intermediates observed during the measurement. However, if the document later studies the end-product, recovered material, post-mortem specimen, or treated material as a distinct identified specimen with its own measurement data, treat that recovered specimen as a separate sample under Step 1.

### `samples[].id`

**Type:** `string`

**Description:** Stable local sample identifier.

**Instructions:**
- Assign IDs in sequence: sample_1, sample_2, sample_3, etc.
- Use this ID consistently when referencing this sample from datasets.

### `samples[].sample_name`

**Type:** `string`

**Description:** Name, label, or concise identifying phrase for the sample.

**Instructions:**
- Follow this decision routine to extract sample_name.
- 1. Use the explicit identifier the document attaches to this sample, e.g. a label, name, composition, formula, material identifier, or concise identifying phrase.
- 2. For a sample-defining variant created under samples Step 4, use the series name plus variant value as printed, e.g. 'LSCNO x=0.05'.
- 3. If the document identifies only the measured group rather than individual specimens (e.g. 'the 50 batches'), use the shortest phrase that identifies the measured group as printed.
- Never use null or an empty string; this field is a required string.

### `samples[].sample_description`

**Type:** `string|null`

**Description:** Short free-text description of sample-defining details not captured by other sample fields.

**Instructions:**
- Use one short sentence based only on the document text.
- Include only sample-defining details not captured by sample_name, chemical_composition, sample_state, sample_mass, or supplier_or_creator.
- Do not include measurement techniques, facilities, measurement results, interpretation, or long synthesis procedures.
- Use null if the structured sample fields already capture the sample sufficiently.

### `samples[].chemical_composition`

**Type:** `string|null`

**Description:** Chemical composition or chemical identifier stated for the sample.

**Instructions:**
- Follow this decision routine to extract chemical_composition.
- 1. Use the explicit composition or formula assigned to this sample, e.g. 'La2CuO4', 'CaCO3'.
- 2. For a composition-series variant, use the formula and variant value as stated in the document, e.g. 'La1-ySryCo1-xNbxO3 x=0.05 y=2x'.
- 3. For a group-level composition series, use the series formula as stated in the document, e.g. 'La1-ySryCo1-xNbxO3'.
- 4. For mixtures, composites, multilayers, alloys, hydrates, isotopically labeled materials, or substrate-supported stacks, preserve the composition phrase as stated, e.g. 'Ni/Ti'.
- 5. If no formula or composition is given but a chemical identifier (such as a CAS number, SMILES, or InChI) is given, use the chemical identifier as stated.
- 6. Do not calculate, expand, simplify, normalize, or convert formulas unless the document explicitly gives that form.
- 7. Do not convert material names, trade names, or abbreviations into formulas using external knowledge.
- 8. Use null if no chemical composition, formula, or chemical identifier is stated for the sample.

### `samples[].sample_state`

**Type:** `string|null`

**Description:** Physical state or form of the sample as written in the document.

**Instructions:**
- Physical state of the sample as initially prepared or supplied for the reported measurements.
- Use short common terms (e.g. 'powder', 'single crystal', 'polycrystalline', 'thin film', 'multilayer', 'nanoparticles', 'glass', 'bulk', 'solution', 'liquid', 'gas').
- If both a general and more specific physical form are stated, use the more specific one, e.g. 'nanoparticles' rather than 'powder', or 'multilayer' rather than 'thin film'.
- Use null if no physical form is stated in the document.

### `samples[].sample_mass`

**Type:** `string|null`

**Description:** Mass of the sample as stated in the document, including value and unit.

**Instructions:**
- Extract only the mass of the sample used for experimental measurement.
- Preserve the value and unit exactly as written in the document, e.g. '5 mg', '0.2 g', '100 µg'.
- Do not include uncertainty notation ('±', about, etc.) in the value.
- Use null if not stated.

### `samples[].supplier_or_creator`

**Type:** `string|null`

**Description:** Short statement of where the sample came from, if explicitly stated in the document.

**Instructions:**
- Follow this decision routine to extract supplier_or_creator.
- 1. If the sample was obtained from an external source, record the source as stated.
- 2. If the document explicitly states the authors prepared the sample, use 'prepared in house'. Explicit attribution means either:
  - (a) first-person wording (`'we synthesized'`, `'prepared in our laboratory'`), or
  - (b) a preparation procedure described in this document with concrete steps (reagents, conditions, processing), not just a method name.
- 3. Use null if not stated.

**Worked example (passive voice with reference to other work):**
arXiv:2104.10884 writes *"Polycrystalline samples of La1−ySryCo1−xNbxO3 … were synthesized by the conventional solid-state method, more details of preparation and characterization can be found in references [13, 43]."* There is no first-person attribution, and the procedural detail is deferred to other references — only the method name is given in this document. Neither (a) nor (b) is met, so `supplier_or_creator = null`.

### `datasets`

**Type:** `list[dataset]`
**Cardinality:** min `0`, max `50`

**Description:** Experimental measurement datasets reported or reanalyzed in the document, grouped by experimental technique and distinct measurement campaign rather than by downstream analysis outputs.

**Instructions:**
- Follow this decision routine to determine dataset entries.
- Step 1: Identify candidate named technique phrases used for reported data in the document's own study.
-   1.1 Exclude phrases mentioned only in background, references, cited work, general setup context, comparison context, or future-work discussion which do not relate to the actual experimental measurements.
-   1.2 For each candidate phrase, remove any part that is only data processing, reconstruction, fitting, refinement, inversion, mapping, clustering, simulation, or interpretation of already-collected measurement data. Also treat suffixes such as '... mapping', '... reconstruction', '... refinement', '... fitting', '... analysis', or '... inversion' as warnings that the named part may be downstream analysis. Keep only experimental measurement technique parts that produced the data, e.g. 'Rietveld refinement of powder X-ray diffraction data' leaves the powder X-ray diffraction measurement technique and treats Rietveld refinement as analysis.
-   1.3 After applying 1.2, classify the remaining phrase: (a) if exactly one experimental measurement technique remains, create one dataset for that technique; (b) if two or more experimental measurement techniques remain, go to Step 2; (c) if no experimental measurement technique remains, do not create a dataset.
- Step 2: For phrases with two or more remaining experimental measurement techniques, decide whether they form one dataset or multiple datasets.
-   2.1 Keep them as one dataset when the document presents them as a single coupled or compound acquisition by either (a) stating they were acquired simultaneously, or (b) consistently reports/interprets them as one inseparable dataset.
-   2.2 Split them into separate datasets when each technique independently produced its own reported measurement data, especially when the document gives separate acquisition conditions, instruments, facilities, or results for each technique.
-   2.3 If you cannot determine whether the techniques are coupled or independently acquired, default to one dataset.
- Step 3: Split the same measurement technique into multiple dataset entries only when the document attaches a qualifier to the technique name and reports the qualified uses separately. The qualifier must indicate a distinct measurement campaign, instrument setup, or acquisition geometry, not merely a variable varied during one campaign. Examples: 'in situ PXRD' vs 'ex situ PXRD' when both are named and reported separately.
- Step 4: Do not create separate dataset entries for parameters varied within one measurement campaign, e.g. temperatures, pressures, magnetic fields, time points, polarizations, detector positions, absorption edges, or nuclei/isotopes. If a Step 3 qualifier turns out to be one of these parameters varied within one measurement campaign, Step 4 takes precedence.
- Step 5: Do not create separate dataset entries solely for different samples, facilities, scan parameters, or downstream analysis outputs. Use sample_refs, facility_refs, dataset_description, and measurement_mode to capture those details.

### `datasets[].id`

**Type:** `string`

**Description:** Stable local dataset identifier for this dataset.

**Instructions:**
- Assign IDs sequentially in order of first appearance in the text: dataset_1, dataset_2, dataset_3, etc.

### `datasets[].dataset_name`

**Type:** `string`

**Description:** Short label identifying this dataset within the document.

**Instructions:**
- Follow this decision routine to determine the dataset name.
- 1. Use the explicit dataset label or name assigned by the document when available, e.g. 'dataset A', 'measurement 1', 'reference measurement'.
- 2. If no explicit dataset label is given, use the technique identified in datasets instructions.
-   - For two or more measurement techniques kept as one dataset under Step 2.1 of the datasets instructions, use the joint term as written (e.g. 'TGA/DSC', 'SAXS/WAXS').
-   - For a phrase where one part is the measurement technique and another part is processing or analysis, use the document phrase as dataset_name when it is the clearest label (e.g. 'PEEM-PIC mapping').
-   - For a qualifier-split entry, include the qualifier as a prefix (e.g. 'in situ PXRD', 'ex situ PXRD').
- 3. Keep the label concise, preferably 2-8 words.

### `datasets[].sample_refs`

**Type:** `list[string]`

**Description:** References to sample identifiers in the samples list for samples measured in this dataset.

**Instructions:**
- Use the exact id values assigned in the samples list (e.g. sample_1, sample_2).
- Include every sample from samples[] that was measured in this dataset.
- Use [] if no samples are associated with this dataset.

### `datasets[].facility_refs`

**Type:** `list[string]`

**Description:** References to facility identifiers in the facilities list for facilities used in this dataset.

**Instructions:**
- Use the exact id values assigned in the facilities list, e.g. facility_1, facility_2.
- Include only facilities explicitly linked to this dataset in the document.
- Use [] if no facility is explicitly associated with this dataset.
- Do not infer facility links from external knowledge.

### `datasets[].dataset_description`

**Type:** `string|null`

**Description:** Brief free-text description of what was measured in this dataset.

**Instructions:**
- Use one sentence based only on the document text.
- Describe the measured quantity, sample context, acquisition context, or key distinguishing details not captured by dataset_name, sample_refs, facility_refs, measurement_technique, measurement_mode, or purpose.
- If the technique was applied with specific scan parameters that do not create separate entries (e.g. absorption edges measured, isotopes probed, temperatures scanned) or with a non-split qualifier ('in situ' mentioned once, with no contrasted counterpart), include that context here.
- Processing or analysis outputs derived from this measurement (e.g. reconstructions, fitted parameters, orientation/PIC maps) may be noted here, but do not form separate datasets.
- Use null if the other dataset fields already identify the dataset sufficiently.

### `datasets[].measurement_technique`

**Type:** `string`

**Description:** The named experimental measurement technique that produced this dataset, as written in the document.

**Instructions:**
- Record the experimental measurement technique identified in datasets instructions.
- For a phrase where one part is the measurement technique and another part is processing or analysis, record only the base measurement technique, e.g. 'powder X-ray diffraction' for 'Rietveld refinement of powder X-ray diffraction data'.
- For two or more measurement techniques kept as one dataset under datasets instructions Step 2.1, record the joint term (e.g. 'TGA/DSC').
- Use the wording from the document; if both a full name and acronym are given, use the full name.
- For a dataset entry created via a qualifier split (Step 3 of the dataset instructions), record the base technique name here without the qualifier (e.g. 'PXRD', not 'in situ PXRD').

### `datasets[].measurement_mode`

**Type:** `list[string]`

**Description:** Major variable or variables varied during this dataset.

**Instructions:**
- Use only the allowed values.
- Assign a scan mode only when the document clearly states or labels the varied acquisition variable, such as an energy range, wavelength series, frequency, angle range, time series, temperature series, pressure series, magnetic-field series, polarization series, position, composition series, or pH series.
- Use multiple modes only when multiple variables are clearly and intentionally varied in the same dataset.
- Use 'Snapshot' only when the document clearly describes a fixed-condition or single-point acquisition and no varied acquisition variable is reported.
- Use [] when the measurement mode cannot be clearly determined from the document text.
- Do not create separate dataset entries only because multiple measurement modes are present in one measurement campaign.
- Do not infer a scan mode from general domain knowledge if the varied acquisition variable is not stated or labeled in the document.

**Constraints:** `allowed_values`: `Snapshot`, `EnergyScan`, `WavelengthScan`, `FrequencyScan`, `AngleScan`, `TimeScan`, `TemperatureScan`, `PressureScan`, `MagneticFieldScan`, `PolarizationScan`, `PositionScan`, `CompositionScan`, `pHScan`, `OtherScan`

### `datasets[].purpose`

**Type:** `string|null`

**Description:** Role of this dataset in the study.

**Instructions:**
- Follow this decision routine to determine the purpose of the dataset.
- 1. Use 'background' for blank, empty-cell, substrate-only, solvent-only, or other measurements used mainly for subtraction or correction.
- 2. Use 'calibration' for reported measurement data used to calibrate an instrument, determine scale/energy/geometry, align a setup, or validate measurement performance.
- 3. Use 'reference' for measurements on a known reference material, comparison sample, standard, or control sample.
- 4. Use 'measurement' for primary or supporting experimental data used in the study.
- 5. Use null if the dataset role cannot be determined from the document.

**Constraints:** `allowed_values`: `measurement`, `reference`, `background`, `calibration`
