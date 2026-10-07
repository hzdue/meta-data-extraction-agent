from __future__ import annotations

import json
import re
import time

from typing import Any

from meta_data_extraction.base.extractor import BaseExtractor
from meta_data_extraction.base.llm import BaseLLM
from meta_data_extraction.data_models import Document, ExtractionResult, Provenance
from meta_data_extraction.utils import (
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_FEWSHOT,
    build_few_shot_prompt,
    build_prompt,
    get_prompt_version,
    get_schema_refs,
)


class LLMExtractor(BaseExtractor):
    def __init__(
        self,
        llm: BaseLLM,
        domain_name: str,
        schemas: list[str],
        max_doc_chars: int = 400_000,
        shot_mode: str = "zero_shot",
        few_shot_examples: list[dict[str, Any]] | None = None,
    ):
        if shot_mode == "few_shot" and not few_shot_examples:
            raise ValueError("shot_mode='few_shot' requires few_shot_examples")
        self.llm = llm
        self.domain_name = domain_name
        self.schemas = schemas
        self.max_doc_chars = max_doc_chars
        self.shot_mode = shot_mode
        self.few_shot_examples = few_shot_examples or []

    def extract(self, document: Document) -> ExtractionResult:
        warnings: list[str] = []

        doc_text = document.text
        if len(doc_text) > self.max_doc_chars:
            doc_text = doc_text[: self.max_doc_chars]
            warnings.append(
                f"Document text truncated from {len(document.text):,} to "
                f"{self.max_doc_chars:,} characters to stay within context limit."
            )

        if self.shot_mode == "few_shot":
            prompt = build_few_shot_prompt(
                schemas=self.schemas,
                document_text=doc_text,
                examples=self.few_shot_examples,
            )
            template_path = PROMPT_TEMPLATE_FEWSHOT
        else:
            prompt = build_prompt(schemas=self.schemas, document_text=doc_text)
            template_path = PROMPT_TEMPLATE

        t0 = time.monotonic()
        response = self.llm.generate(prompt)
        latency_ms = (time.monotonic() - t0) * 1000

        if response.finish_reason == "length":
            warnings.append(
                "LLM response was cut off at the token limit (finish_reason='length'). "
                "JSON output may be incomplete. Consider increasing max_tokens."
            )

        output, parse_warnings = self._parse_output(response.content)
        warnings.extend(parse_warnings)

        return ExtractionResult(
            document_id=document.id,
            domain_name=self.domain_name,
            output=output,
            raw_output=response.content,
            provenance=Provenance(
                prompt_version=get_prompt_version(template_path),
                schemas=get_schema_refs(self.schemas),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=round(latency_ms, 2),
            ),
            warnings=warnings,
        )

    def _parse_output(self, raw: str) -> tuple[dict, list[str]]:
        warnings: list[str] = []
        text = raw.strip()

        # Strip markdown code fences if present.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            output = json.loads(text)
            if not isinstance(output, dict):
                warnings.append("LLM output parsed but is not a JSON object; using empty output.")
                return {}, warnings
            return output, warnings
        except json.JSONDecodeError:
            pass

        # Fallback: try the span from the first opening brace to the last closing brace.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                output = json.loads(match.group())
                if isinstance(output, dict):
                    warnings.append(
                        "JSON extracted from partial response after initial parse failure; "
                        "output may be incomplete."
                    )
                    return output, warnings
            except json.JSONDecodeError:
                pass

        warnings.append("JSON parse failed after fallback extraction; using empty output.")
        return {}, warnings
