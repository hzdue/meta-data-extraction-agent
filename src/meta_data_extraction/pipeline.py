from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from meta_data_extraction.app_config import AppConfig
from meta_data_extraction.base.extractor import BaseExtractor
from meta_data_extraction.base.parser import BaseParser
from meta_data_extraction.base.tracker import BaseTracker
from meta_data_extraction.base.validator import BaseValidator
from meta_data_extraction.components.extractor import LLMExtractor
from meta_data_extraction.components.llm import OpenAICompatibleLLM
from meta_data_extraction.components.parser import PdfParser
from meta_data_extraction.components.tracker import LocalTracker
from meta_data_extraction.components.validator import SimpleValidator
from meta_data_extraction.data_models import ExtractionResult
from meta_data_extraction.utils import load_few_shot_examples


@dataclass
class Pipeline:
    domain_name: str
    schemas: list[str]
    parser: BaseParser
    extractor: BaseExtractor
    validator: BaseValidator
    tracker: BaseTracker | None = None
    config_name: str | None = None  # stem of the config file, stored in provenance

    @classmethod
    def from_config(cls, config: AppConfig, config_path: Path | str | None = None) -> "Pipeline":
        # Load .env without overriding existing environment variables.
        load_dotenv()

        if config.parser.type != "pdf":
            raise ValueError(
                f"Unsupported parser type {config.parser.type!r}. "
                "Currently only 'pdf' is supported."
            )
        parser = PdfParser()

        if config.llm.provider == "openai_compatible":
            llm = OpenAICompatibleLLM(
                model_name=config.llm.model,
                api_key_env=config.llm.api_key_env,
                base_url=config.llm.base_url,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                max_tokens_param=config.llm.max_tokens_param,
                reasoning_effort=config.llm.reasoning_effort,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider {config.llm.provider!r}. "
                "Supported: 'openai_compatible'."
            )

        # Use the same loaded exemplars for every document in this pipeline.
        few_shot_examples = None
        if config.extractor.shot_mode == "few_shot":
            assert config.extractor.few_shot is not None
            few_shot_examples = load_few_shot_examples(
                config.extractor.few_shot.example_ids,
                config.extractor.few_shot.examples_dir,
            )

        extractor = LLMExtractor(
            llm,
            domain_name=config.domain_name,
            schemas=config.schemas,
            max_doc_chars=config.llm.max_doc_chars,
            shot_mode=config.extractor.shot_mode,
            few_shot_examples=few_shot_examples,
        )
        validator = SimpleValidator()
        tracker = LocalTracker(output_dir=config.tracker.output_dir)

        return cls(
            domain_name=config.domain_name,
            schemas=config.schemas,
            parser=parser,
            extractor=extractor,
            validator=validator,
            tracker=tracker,
            config_name=Path(config_path).stem if config_path else None,
        )

    def run(self, path: Path | str) -> ExtractionResult:
        document = self.parser.parse(Path(path))
        result = self.extractor.extract(document=document)
        result = self.validator.validate(result=result, schemas=self.schemas)

        if result.provenance and self.config_name:
            result.provenance.config_name = self.config_name

        if self.tracker:
            self.tracker.log_run(result)

        return result
