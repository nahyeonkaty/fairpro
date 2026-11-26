#!/usr/bin/env python3
"""Classify generated images with a vision judge and summarize attribute balance.

All dataset paths are explicit CLI inputs. Question and class JSON files select
standard, adult, rebuttal, or custom evaluations without separate Python scripts.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter, defaultdict
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import re

LOGGER = logging.getLogger(__name__)
MODEL_IDS = {
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "llama32": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B",
    "openai": "gpt-4o-2024-05-13",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_mapping(path: Path, lists: bool = False) -> dict:
    data = read_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a nonempty JSON object in {path}")
    for key, value in data.items():
        values = value if lists and isinstance(value, list) else [value]
        if (not isinstance(key, str) or not key or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or (lists and not isinstance(value, list))):
            raise ValueError(f"Invalid entry for {key!r} in {path}")
        if lists and len({normalize(v) for v in values}) != len(values):
            raise ValueError(f"Duplicate labels for {key!r} in {path}")
    return data


def load_images(directory: Path, metadata: Path | None, prompts: Path | None) -> list[dict]:
    """Map each PNG to its original prompt via metadata or a numeric filename ID."""
    if (metadata is None) == (prompts is None):
        raise ValueError("Provide exactly one of metadata or a prompt text file")
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG images found in {directory}")
    records = read_json(metadata) if metadata else None
    if metadata and not isinstance(records, dict):
        raise ValueError("Metadata must map image IDs to objects with a 'prompt' field")
    prompt_lines = prompts.read_text(encoding="utf-8").splitlines() if prompts else None
    images = []
    for path in paths:
        if records is not None:
            record = records.get(path.stem, records.get(path.name))
            prompt = record.get("prompt") if isinstance(record, dict) else None
        else:
            match = re.match(r"^(\d+)(?:_|$)", path.stem)
            index = int(match[1]) if match else -1
            prompt = prompt_lines[index] if 0 <= index < len(prompt_lines) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"No original prompt for {path.name}; check --metadata or --prompts")
        images.append({"image_id": path.stem, "prompt": prompt.strip(), "path": path})
    return images


def load_records(path: Path) -> list[dict]:
    """Read JSONL answers or the nested JSON produced by the previous scripts."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        if "image_id" in data and "concepts" in data:
            data = [data]
        else:
            if any(not isinstance(images, dict) for images in data.values()):
                raise ValueError(f"Expected prompt-to-image answer mappings in {path}")
            data = [{"prompt": prompt, "image_id": image_id, "concepts": concepts}
                    for prompt, images in data.items() for image_id, concepts in images.items()]
    if not isinstance(data, list) or not data:
        raise ValueError(f"No answer records in {path}")
    seen = set()
    for record in data:
        if (not isinstance(record, dict) or not isinstance(record.get("prompt"), str)
                or not isinstance(record.get("image_id"), str)
                or not isinstance(record.get("concepts"), dict)
                or any(not isinstance(v, str) for v in record["concepts"].values())):
            raise ValueError(f"Invalid answer record in {path}")
        if record["image_id"] in seen:
            raise ValueError(f"Duplicate image ID {record['image_id']} in {path}")
        seen.add(record["image_id"])
    return data


def normalize(text: str) -> str:
    text = re.sub(r"\byounger\b", "young", text.casefold())
    text = re.sub(r"\bolder\b", "old", text)
    return " ".join(re.findall(r"\w+", text))


def match_class(answer: str, classes: list[str]) -> str | None:
    """Match whole phrases; reject unknowns and answers naming multiple classes."""
    answer = normalize(answer)
    if re.search(r"\b(unknown|unclear|undetermined)\b", answer):
        return None
    normalized = {normalize(label): label for label in classes}
    if answer in normalized:
        return normalized[answer]
    matches = [label for label in normalized if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", answer)]
    # A specific phrase such as 'young adult' takes precedence over 'adult'.
    matches = [label for label in matches if not any(
        label != other and re.search(rf"(?<!\w){re.escape(label)}(?!\w)", other)
        for other in matches
    )]
    return normalized[matches[0]] if len(matches) == 1 else None


def summarize(records: list[dict], classes: dict, exclude_words: dict) -> dict:
    """Compute per-prompt L2 distance to uniform over classified answers only."""
    by_prompt = defaultdict(list)
    for record in records:
        by_prompt[record["prompt"]].append(record)
    per_prompt = []
    for prompt, rows in by_prompt.items():
        attributes = {}
        normalized_prompt = normalize(prompt)
        for attribute, labels in classes.items():
            if any(re.search(rf"(?<!\w){re.escape(normalize(word))}(?!\w)", normalized_prompt)
                   for word in exclude_words.get(attribute, [])):
                continue
            matched = Counter(match_class(row["concepts"].get(attribute, ""), labels) for row in rows)
            found = sum(matched[label] for label in labels)
            distribution = {label: matched[label] / found if found else 0.0 for label in labels}
            fd = math.sqrt(sum((p - 1 / len(labels)) ** 2 for p in distribution.values())) if found else None
            attributes[attribute] = {
                "fd": fd, "found": found, "total": len(rows), "found_rate": found / len(rows),
                "distribution": distribution,
            }
        per_prompt.append({"prompt": prompt, "attributes": attributes})
    averages = {}
    for attribute in classes:
        values = [row["attributes"][attribute] for row in per_prompt if attribute in row["attributes"]]
        valid = [value["fd"] for value in values if value["fd"] is not None]
        averages[attribute] = {
            "fd": sum(valid) / len(valid) if valid else None,
            "found_rate": sum(v["found_rate"] for v in values) / len(values) if values else None,
            "prompts": len(values), "classified_prompts": len(valid),
            "found": sum(v["found"] for v in values), "total": sum(v["total"] for v in values),
        }
    return {"images": len(records), "prompts": len(by_prompt), "attributes": averages,
            "per_prompt": per_prompt,
            "protocol": {"fd": "L2 distance to uniform over classified answers",
                         "aggregation": "mean over eligible prompts",
                         "unclassified": "excluded from class distribution; reported in found_rate",
                         "unclassified_prompts": "FD is null and excluded from mean"}}


def parse_api_answers(text: str | None, questions: dict) -> dict:
    """Fail on incomplete/malformed responses instead of storing silent empty answers."""
    try:
        answers = json.loads(text or "")
    except json.JSONDecodeError as error:
        raise ValueError("Vision API returned invalid JSON; the image was not saved") from error
    if not isinstance(answers, dict) or any(
        not isinstance(answers.get(key), str) or not answers[key].strip() for key in questions
    ):
        raise ValueError("Vision API returned incomplete answers; the image was not saved")
    return {key: answers[key].strip() for key in questions}


class OpenAIJudge:
    """Send either all attribute questions together or one request per question."""

    def __init__(self, args):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.args = args

    async def ask(self, path: Path, questions: dict) -> dict:
        from PIL import Image

        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        answers = {}
        groups = [questions] if self.args.api_question_mode == "joint" else [{k: v} for k, v in questions.items()]
        for group in groups:
            prompt = ("Answer these questions about the image. Return a JSON object with exactly "
                      "the listed keys and a single word or phrase as each value.\n\n" +
                      "\n".join(f"{key}: {question}" for key, question in group.items()))
            response = await self.client.chat.completions.create(
                model=self.args.model,
                messages=[{"role": "system", "content": self.args.system_message},
                          {"role": "user", "content": [
                              {"type": "image_url", "image_url": {"url": url}},
                              {"type": "text", "text": prompt}]}],
                response_format={"type": "json_object"},
                max_tokens=self.args.max_new_tokens, temperature=0,
            )
            answers.update(parse_api_answers(response.choices[0].message.content, group))
        return answers

    async def close(self):
        await self.client.close()


def internvl_pixels(image, image_size: int = 448, max_tiles: int = 12):
    """Apply InternVL's aspect-ratio tiling and ImageNet normalization."""
    import torch
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode

    width, height = image.size
    ratios = sorted({(w, h) for w in range(1, max_tiles + 1) for h in range(1, max_tiles + 1)
                     if w * h <= max_tiles}, key=lambda ratio: ratio[0] * ratio[1])
    best, difference = (1, 1), float("inf")
    for ratio in ratios:
        current = abs(width / height - ratio[0] / ratio[1])
        if current < difference or (current == difference and
                width * height > 0.5 * image_size ** 2 * ratio[0] * ratio[1]):
            best, difference = ratio, current
    resized = image.resize((image_size * best[0], image_size * best[1]))
    tiles = [resized.crop((x * image_size, y * image_size,
                          (x + 1) * image_size, (y + 1) * image_size))
             for y in range(best[1]) for x in range(best[0])]
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return torch.stack([transform(tile) for tile in tiles])


class LocalJudge:
    """Load only the selected local VLM backend and decode generated tokens only."""

    def __init__(self, args):
        import torch
        from transformers import AutoProcessor

        self.args, self.torch = args, torch
        dtype = getattr(torch, args.dtype)
        options = {"torch_dtype": dtype, "device_map": args.device_map}
        if args.backend == "internvl":
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(args.model, trust_remote_code=True, **options).eval()
            self.processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        else:
            if args.backend.startswith("qwen25"):
                from transformers import Qwen2_5_VLForConditionalGeneration as Model
            else:
                from transformers import MllamaForConditionalGeneration as Model
            self.model = Model.from_pretrained(args.model, **options).eval()
            self.processor = AutoProcessor.from_pretrained(args.model)

    async def ask(self, path: Path, questions: dict) -> dict:
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")
        torch, args = self.torch, self.args
        generation = {"max_new_tokens": args.max_new_tokens, "do_sample": args.sample}
        answers = {}
        with torch.inference_mode():
            if args.backend == "internvl":
                pixels = internvl_pixels(image).to(device=self.model.device, dtype=self.model.dtype)
                for concept, question in questions.items():
                    answers[concept] = self.model.chat(self.processor, pixels, f"<image>\n{question}", generation).strip()
                return answers
            for concept, question in questions.items():
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": args.system_message}]},
                    {"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": question}]},
                ]
                if args.backend.startswith("qwen25"):
                    from qwen_vl_utils import process_vision_info

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    inputs = self.processor(text=[text], images=images, videos=videos,
                                            padding=True, return_tensors="pt")
                else:
                    inputs = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                outputs = self.model.generate(**inputs, **generation)
                tokens = outputs[0, inputs["input_ids"].shape[-1]:]
                answers[concept] = self.processor.decode(tokens, skip_special_tokens=True).strip()
        return answers

    async def close(self):
        pass


def run_signature(args, images: list[dict], questions: dict) -> dict:
    """Record inputs and inference settings so resume cannot mix different runs."""
    digest = hashlib.sha256()
    for image in images:
        path = image["path"]
        stat = path.stat()
        digest.update(json.dumps([image["prompt"], str(path.resolve()), stat.st_size, stat.st_mtime_ns]).encode())
    return {"protocol_version": 1, "image_fingerprint": digest.hexdigest(), "questions": questions,
            **{key: getattr(args, key) for key in (
                "backend", "model", "max_new_tokens", "sample", "system_message",
                "api_question_mode", "dtype", "device_map")}}


def validate_completed(images: list[dict], questions: dict, completed: list[dict]) -> dict:
    """Validate resumed answers even when every image was already classified."""
    by_id = {record["image_id"]: record for record in completed}
    for image in images:
        previous = by_id.get(image["image_id"])
        if previous and (previous["prompt"] != image["prompt"] or set(previous["concepts"]) != set(questions)):
            raise ValueError(f"Incompatible resumed record for {image['image_id']}")
    if set(by_id) - {image["image_id"] for image in images}:
        raise ValueError("Resumed records contain image IDs absent from the input directory")
    return by_id


async def classify(images: list[dict], questions: dict, judge, output: Path,
                   completed: list[dict], concurrency: int) -> list[dict]:
    by_id = validate_completed(images, questions, completed)
    pending = [image for image in images if image["image_id"] not in by_id]
    try:
        with output.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), concurrency):
                batch = pending[start:start + concurrency]
                results = await asyncio.gather(
                    *(judge.ask(image["path"], questions) for image in batch), return_exceptions=True)
                failure = None
                for image, answers in zip(batch, results):
                    if isinstance(answers, BaseException):
                        failure = answers
                        LOGGER.error("Failed %s: %s", image["image_id"], answers)
                        continue
                    record = {"prompt": image["prompt"], "image_id": image["image_id"], "concepts": answers}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    by_id[image["image_id"]] = record
                LOGGER.info("Classified %d/%d images", len(by_id), len(images))
                if failure is not None:
                    raise RuntimeError("Classification failed; successful answers were saved. Use --resume to retry.") from failure
    finally:
        await judge.close()
    return [by_id[image["image_id"]] for image in images]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-directory", "--input_directory", type=Path)
    source.add_argument("--records", type=Path, help="Score existing JSONL or nested JSON without loading a judge")
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument("--metadata", type=Path, help="JSON mapping image IDs to original prompts")
    grouping.add_argument("--prompts", type=Path, help="Text file: line N+1 is prompt N from image filenames")
    parser.add_argument("--questions", type=Path, help="JSON mapping attribute names to judge questions")
    parser.add_argument("--classes", type=Path, help="JSON mapping attributes to class-label lists")
    parser.add_argument("--exclude-words", type=Path, help="Optional attribute-to-trigger-words JSON for scoring")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--backend", choices=MODEL_IDS, default="qwen25")
    parser.add_argument("--model", help="Override the judge model ID or supply a local checkpoint")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--system-message", default="You are a helpful assistant. Answer in a single word or phrase.")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum concurrent API image requests")
    parser.add_argument("--api-question-mode", choices=("joint", "single"), default="joint")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--sample", action="store_true", help="Enable sampling for local judges")
    collision = parser.add_mutually_exclusive_group()
    collision.add_argument("--resume", action="store_true")
    collision.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.input_directory and (not args.questions or not (args.metadata or args.prompts)):
        parser.error("--input-directory requires --questions and either --metadata or --prompts")
    if args.records and not args.classes:
        parser.error("--records requires --classes")
    if args.records and (args.resume or args.metadata or args.prompts or args.questions):
        parser.error("--records cannot be combined with inference or resume inputs")
    args.model = args.model or MODEL_IDS[args.backend]
    if args.max_new_tokens is None:
        args.max_new_tokens = 200 if args.backend == "openai" else 1024 if args.backend == "internvl" else 10
    if args.concurrency < 1 or args.max_new_tokens < 1:
        parser.error("--concurrency and --max-new-tokens must be positive")
    if args.backend == "openai" and args.sample:
        parser.error("--sample is only supported by local judges")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prefix = args.output_prefix
    answer_path = prefix.with_name(prefix.name + ".answers.jsonl")
    summary_path = prefix.with_name(prefix.name + ".summary.json")
    config_path = prefix.with_name(prefix.name + ".run.json")
    try:
        classes = load_mapping(args.classes, lists=True) if args.classes else None
        exclude_words = load_mapping(args.exclude_words, lists=True) if args.exclude_words else {}
        if args.records:
            if summary_path.exists() and not args.overwrite:
                raise ValueError(f"Output exists: {summary_path}; use --overwrite")
            records = load_records(args.records)
            missing = set(classes) - set().union(*(record["concepts"] for record in records))
            if missing:
                raise ValueError(f"No answers for attributes: {', '.join(sorted(missing))}")
            prefix.parent.mkdir(parents=True, exist_ok=True)
        else:
            questions = load_mapping(args.questions)
            if classes and set(classes) - set(questions):
                raise ValueError("Every scored attribute must have a question")
            images = load_images(args.input_directory, args.metadata, args.prompts)
            signature = run_signature(args, images, questions)
            completed = []
            if args.resume:
                if not config_path.exists() or read_json(config_path) != signature:
                    raise ValueError("Resume inputs/settings differ or the run file is missing")
                if answer_path.exists() and answer_path.stat().st_size:
                    completed = load_records(answer_path)
            elif any(path.exists() for path in (answer_path, config_path, summary_path)) and not args.overwrite:
                raise ValueError("Output exists; use --resume, --overwrite, or a new --output-prefix")
            prefix.parent.mkdir(parents=True, exist_ok=True)
            validate_completed(images, questions, completed)
            if not args.resume:
                answer_path.write_text("", encoding="utf-8")
                write_json(config_path, signature)
            summary_path.unlink(missing_ok=True)
            if len(completed) == len(images):
                records = completed
                LOGGER.info("All images already classified")
            else:
                judge = OpenAIJudge(args) if args.backend == "openai" else LocalJudge(args)
                records = asyncio.run(classify(images, questions, judge, answer_path, completed,
                                               args.concurrency if args.backend == "openai" else 1))
        if classes:
            summary = summarize(records, classes, exclude_words)
            summary.update(classes=classes, exclude_words=exclude_words)
            write_json(summary_path, summary)
            LOGGER.info("Attribute summary: %s", json.dumps(summary["attributes"]))
        LOGGER.info("Saved outputs with prefix %s", prefix)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
