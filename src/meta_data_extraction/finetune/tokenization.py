"""Count real chat-template sequences without loading a model or truncating."""
from __future__ import annotations

from importlib.metadata import version
import re

from .parsing import text_hash


def load_tokenizer(config: dict, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    if config["model_id"] != "Qwen/Qwen3.5-4B" or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pin the official Qwen3.5-4B model to a full commit hash")
    if config["enable_thinking"] is not False:
        raise ValueError("This pilot requires thinking disabled")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if not tokenizer.chat_template or tokenizer.eos_token != "<|im_end|>":
        raise ValueError("Unexpected tokenizer/chat-template contract")
    identity = {
        "model_id": config["model_id"], "revision": config["revision"],
        "class": type(tokenizer).__name__, "transformers": version("transformers"),
        "tokenizers": version("tokenizers"), "enable_thinking": False,
        "chat_template_sha256": text_hash(tokenizer.chat_template),
        "backend_sha256": text_hash(tokenizer.backend_tokenizer.to_str()),
        "model_max_length": tokenizer.model_max_length, "eos_token_id": tokenizer.eos_token_id,
    }
    return tokenizer, identity


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def encode_record(tokenizer, record: dict) -> tuple[list[int], int, int]:
    """Return full IDs, target start, target+EOS end (exclusive).

    Empty non-thinking scaffolding belongs to the prompt. The template's trailing
    newline remains in the full sequence, outside the supervised target+EOS span.
    Step 3 must use these boundaries rather than re-applying an implicit template.
    """
    prefix = render_prompt(tokenizer, record["prompt"])
    rendered = tokenizer.apply_chat_template(
        record["prompt"] + record["completion"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    if not rendered.startswith(prefix):
        raise ValueError("Completed chat does not share the inference prompt prefix")
    ids = tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False, truncation=False)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer boundary changed between prompt and completed chat")
    eos_positions = [i for i in range(len(prompt_ids), len(ids)) if ids[i] == tokenizer.eos_token_id]
    if len(eos_positions) != 1:
        raise ValueError("Expected exactly one assistant end marker")
    return ids, len(prompt_ids), eos_positions[0] + 1


def record_lengths(tokenizer, record: dict, cap: int) -> dict:
    ids, start, end = encode_record(tokenizer, record)
    return {
        "example_id": record["example_id"], "paper_id": record["paper_id"],
        "split": record["split"], "task": record["task"],
        "prompt_tokens": start, "target_tokens": end - start - 1,
        "supervised_tokens": end - start, "post_eos_tokens": len(ids) - end,
        "sequence_tokens": len(ids), "fits": len(ids) <= cap,
    }


def inference_lengths(tokenizer, paper_id: str, split: str, prompt: str,
                      cap: int, max_new_tokens: int) -> dict:
    rendered = render_prompt(tokenizer, [{"role": "user", "content": prompt}])
    count = len(tokenizer.encode(rendered, add_special_tokens=False, truncation=False))
    return {"paper_id": paper_id, "split": split, "prompt_tokens": count,
            "reserved_output_tokens": max_new_tokens, "required_tokens": count + max_new_tokens,
            "fits": count + max_new_tokens <= cap}
