#!/usr/bin/env python3
"""Measure within-prompt image diversity with CLIP cosine similarity and LPIPS.

Evaluate every distinct unordered image pair within each prompt. Images can be
identified by <prompt>[_sp<N>]_seed<N>.png filenames or a metadata JSON mapping
image IDs to objects with a "prompt" field. See README.md for examples.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations, islice
import json
import logging
from pathlib import Path
import re
import time

IMAGE_PATTERN = re.compile(r"(?P<prompt>\d+)(?:_sp\d+)?_seed(?P<seed>\d+)")
LOGGER = logging.getLogger(__name__)
PROTOCOL = {
    "clip_model": "OpenAI ViT-B/32",
    "lpips_net": "alex",
    "lpips_version": "0.1",
    "lpips_size": [224, 224],
    "lpips_range": [-1, 1],
    "self_pairs": False,
    "pair_sampling": "all_unordered",
    "aggregation": "mean_over_pairs",
}


def discover_groups(directory: Path, metadata: Path | None = None) -> dict[str, list[Path]]:
    """Group PNGs by prompt, rejecting ambiguous or incomplete image groups."""
    images = sorted(directory.glob("*.png"))
    if not images:
        raise ValueError(f"No PNG images found in {directory}")
    groups = defaultdict(list)
    if metadata is not None:
        records = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(records, dict):
            raise ValueError("Metadata must map image IDs to objects with a 'prompt' field")
        for image in images:
            record = records.get(image.stem, records.get(image.name))
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                raise ValueError(f"Missing string 'prompt' in metadata for {image.name}")
            groups[record["prompt"]].append(image)
    else:
        seeds = defaultdict(dict)
        for image in images:
            match = IMAGE_PATTERN.fullmatch(image.stem)
            if match is None:
                raise ValueError(f"Unrecognized filename {image.name}; provide --metadata "
                                 "or use <prompt>[_sp<N>]_seed<N>.png")
            prompt, seed = match["prompt"], int(match["seed"])
            if seed in seeds[prompt]:
                raise ValueError(f"Duplicate seed {seed} for prompt {prompt} in {directory}")
            seeds[prompt][seed] = image
        groups = {prompt: [paths[seed] for seed in sorted(paths)]
                  for prompt, paths in seeds.items()}
    singletons = [prompt for prompt, paths in groups.items() if len(paths) < 2]
    if singletons:
        raise ValueError(f"Each prompt needs at least two images; found {len(singletons)} "
                         f"singleton groups in {directory}")
    return dict(sorted(groups.items()))


def input_fingerprint(groups: dict[str, list[Path]]) -> str:
    """Detect changes to image grouping, file sizes, or modification times on resume."""
    digest = hashlib.sha256()
    for prompt, paths in groups.items():
        for path in paths:
            stat = path.stat()
            record = [prompt, str(path.resolve()), stat.st_size, stat.st_mtime_ns]
            digest.update((json.dumps(record) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def is_complete(summary_path: Path, pair_path: Path, fingerprint: str) -> bool:
    """Only reuse completed outputs from matching inputs and metric settings."""
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (not pair_path.exists() or summary.get("input_fingerprint") != fingerprint
            or any(summary.get(key) != value for key, value in PROTOCOL.items())):
        raise ValueError(f"Existing output is incompatible with this run: {summary_path}. "
                         "Choose a new output directory or pass --overwrite.")
    return True


class DiversityMetrics:
    """Load the two networks once and reuse image encodings within each prompt."""

    def __init__(self, device: str, batch_size: int, memory_fraction: float | None):
        import clip
        import lpips
        import torch
        import torchvision.transforms as transforms

        self.torch = torch
        self.batch_size = batch_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available")
            cuda_device = torch.device(self.device)
            if cuda_device.index is None:
                self.device = f"cuda:{torch.cuda.current_device()}"
            if memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip.eval()
        self.lpips = lpips.LPIPS(net="alex", version="0.1").to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
        ])

    def pairs(self, paths: list[Path]):
        """Yield (image1, image2, CLIP similarity, LPIPS) in bounded pair batches."""
        from PIL import Image
        import torch.nn.functional as functional

        torch = self.torch
        # Keep only resized tensors, rather than full-resolution decoded images.
        clip_inputs, lpips_inputs = [], []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                clip_inputs.append(self.preprocess(rgb))
                lpips_inputs.append(self.transform(rgb))
        with torch.inference_mode():
            features = []
            for start in range(0, len(paths), self.batch_size):
                batch = torch.stack(clip_inputs[start:start + self.batch_size]).to(self.device)
                features.append(self.clip.encode_image(batch).float())
            features = functional.normalize(torch.cat(features), dim=-1)
            # LPIPS expects [-1, 1], whereas ToTensor produces [0, 1].
            pixels = torch.stack(lpips_inputs) * 2 - 1
            pair_iterator = combinations(range(len(paths)), 2)
            while batch_pairs := list(islice(pair_iterator, self.batch_size)):
                left, right = map(list, zip(*batch_pairs))
                similarities = (features[left] * features[right]).sum(-1).cpu().tolist()
                distances = self.lpips(pixels[left].to(self.device), pixels[right].to(self.device))
                for (a, b), similarity, distance in zip(
                    batch_pairs, similarities, distances.flatten().cpu().tolist()
                ):
                    yield paths[a].name, paths[b].name, similarity, distance


def evaluate(groups: dict[str, list[Path]], metrics: DiversityMetrics,
             output_prefix: Path, labels: dict, fingerprint: str) -> dict:
    """Stream pair records and publish a completion summary only after success."""
    started = time.monotonic()
    pair_path = output_prefix.with_name(output_prefix.name + ".pairs.jsonl")
    temporary = pair_path.with_name(pair_path.name + ".tmp")
    count, clip_total, lpips_total = 0, 0.0, 0.0
    with temporary.open("w", encoding="utf-8") as output:
        for index, (prompt, paths) in enumerate(groups.items(), start=1):
            for image1, image2, similarity, distance in metrics.pairs(paths):
                output.write(json.dumps({
                    "prompt_id": prompt, "image1": image1, "image2": image2,
                    "clip_sim_pair": similarity, "lpips": distance,
                }) + "\n")
                count += 1
                clip_total += similarity
                lpips_total += distance
            if index % 16 == 0 or index == len(groups):
                LOGGER.info("%s: %d/%d prompts (%.1fs)", output_prefix.name,
                            index, len(groups), time.monotonic() - started)
    temporary.replace(pair_path)
    pair_counts = {len(paths) * (len(paths) - 1) // 2 for paths in groups.values()}
    summary = {
        **labels, **PROTOCOL, "input_fingerprint": fingerprint,
        "prompts": len(groups), "images": sum(map(len, groups.values())),
        "pairs": count, "pairs_per_prompt": pair_counts.pop() if len(pair_counts) == 1 else None,
        "clip_sim_pair": clip_total / count, "lpips": lpips_total / count,
        "seconds": time.monotonic() - started,
    }
    write_json(output_prefix.with_name(output_prefix.name + ".summary.json"), summary)
    LOGGER.info("%s: CLIP=%.4f LPIPS=%.4f (%d pairs)",
                output_prefix.name, summary["clip_sim_pair"], summary["lpips"], count)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", "--input_directory", type=Path, required=True,
                        help="Directory containing the PNG images to evaluate")
    parser.add_argument("--metadata", type=Path,
                        help="JSON mapping image IDs to prompt records (single-directory mode)")
    parser.add_argument("--save-directory", "--save_directory", type=Path, default=Path("results"))
    parser.add_argument("--name", default="diversity", help="Output filename prefix in single-directory mode")
    parser.add_argument("--model-label", help="Optional model label recorded in the summary")
    parser.add_argument("--variant", help="Optional method/baseline label recorded in the summary")
    parser.add_argument("--level", help="Optional prompt-level label recorded in the summary")
    parser.add_argument("--expected-prompts", type=int, help="Require this many prompt groups")
    parser.add_argument("--expected-images-per-prompt", type=int, help="Require this many images per group")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cpu, cuda, cuda:1")
    parser.add_argument("--batch-size", type=int, default=64, help="Maximum images or pairs per GPU batch")
    parser.add_argument("--memory-fraction", type=float, help="Optional CUDA allocator memory fraction (0, 1]")
    parser.add_argument("--threads", type=int, default=2, help="Number of PyTorch CPU threads")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and replace existing outputs")
    args = parser.parse_args(argv)
    if any(value is not None and value < 1 for value in
           (args.expected_prompts, args.expected_images_per_prompt)):
        parser.error("Expected counts must be positive")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1]")
    if not args.name or args.name in (".", "..") or Path(args.name).name != args.name:
        parser.error("--name must be a filename prefix, without directory components")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        groups = discover_groups(args.input_directory, args.metadata)
        if args.expected_prompts is not None and len(groups) != args.expected_prompts:
            raise ValueError(f"Expected {args.expected_prompts} prompts, found {len(groups)}")
        if args.expected_images_per_prompt is not None and any(
            len(paths) != args.expected_images_per_prompt for paths in groups.values()
        ):
            raise ValueError(f"Expected {args.expected_images_per_prompt} images per prompt")
        prefix = args.save_directory / args.name
        fingerprint = input_fingerprint(groups)
        labels = {"input_directory": str(args.input_directory.resolve())}
        for key, value in (("model", args.model_label), ("mode", args.variant), ("level", args.level)):
            if value is not None:
                labels[key] = value
        summary_path = prefix.with_name(prefix.name + ".summary.json")
        pair_path = prefix.with_name(prefix.name + ".pairs.jsonl")
        if not args.overwrite and is_complete(summary_path, pair_path, fingerprint):
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in labels.items()):
                raise ValueError("Existing summary labels differ; use --overwrite or a new --name")
            LOGGER.info("Skipping complete %s", args.name)
            return
        import torch

        torch.set_num_threads(args.threads)
        metrics = DiversityMetrics(args.device, args.batch_size, args.memory_fraction)
        args.save_directory.mkdir(parents=True, exist_ok=True)
        # Invalidate completion before overwriting so interrupted runs are not reused.
        summary_path.unlink(missing_ok=True)
        evaluate(groups, metrics, prefix, labels, fingerprint)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
