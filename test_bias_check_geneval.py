#!/usr/bin/env python3
"""Test script to verify bias check functionality on geneval prompts.

This script tests the check_prompt_for_bias method with both QwenImage and SANA
pipelines using geneval evaluation prompts.

Usage:
    CUDA_VISIBLE_DEVICES=6 python test_bias_check_geneval.py --model sana
    CUDA_VISIBLE_DEVICES=6 python test_bias_check_geneval.py --model qwenimage
    CUDA_VISIBLE_DEVICES=6 python test_bias_check_geneval.py --model both
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_geneval_prompts(filepath: str) -> list[dict]:
    """Load prompts from geneval JSONL file."""
    prompts = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def test_sana_bias_check(prompts: list[dict], output_file: str | None = None):
    """Test bias check with SANA pipeline."""
    from fairpro import FairProSanaPipeline

    logger.info("Loading SANA pipeline...")
    pipe = FairProSanaPipeline.from_pretrained(
        "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    logger.info("Enabling FairPro...")
    pipe.enable_fairpro()

    results = []
    yes_count = 0
    no_count = 0

    logger.info(f"Testing {len(prompts)} prompts...")
    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        tag = item.get("tag", "unknown")

        try:
            has_bias = pipe.check_prompt_for_bias(prompt)
            result = "yes" if has_bias else "no"
        except Exception as e:
            result = f"error: {e}"
            logger.error(f"Error checking prompt: {prompt[:50]}... - {e}")

        if result == "yes":
            yes_count += 1
        elif result == "no":
            no_count += 1

        results.append(
            {
                "prompt": prompt,
                "tag": tag,
                "bias_check": result,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(prompts)} (yes: {yes_count}, no: {no_count})"
            )

    # Print summary
    print("\n" + "=" * 70)
    print("SANA Bias Check Results Summary")
    print("=" * 70)
    print(f"Total prompts: {len(prompts)}")
    print(
        f"Yes (could lead to bias): {yes_count} ({yes_count / len(prompts) * 100:.1f}%)"
    )
    print(f"No (no bias concern): {no_count} ({no_count / len(prompts) * 100:.1f}%)")

    # Print by tag
    tag_results = {}
    for r in results:
        tag = r["tag"]
        if tag not in tag_results:
            tag_results[tag] = {"yes": 0, "no": 0}
        if r["bias_check"] == "yes":
            tag_results[tag]["yes"] += 1
        elif r["bias_check"] == "no":
            tag_results[tag]["no"] += 1

    print("\nBy category:")
    for tag, counts in sorted(tag_results.items()):
        total = counts["yes"] + counts["no"]
        print(
            f"  {tag}: yes={counts['yes']}, no={counts['no']} ({counts['yes'] / total * 100:.0f}% yes)"
        )

    # Save results to file
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    # Cleanup
    pipe.disable_fairpro()
    del pipe
    torch.cuda.empty_cache()

    return results


def test_qwenimage_bias_check(prompts: list[dict], output_file: str | None = None):
    """Test bias check with QwenImage pipeline."""
    from fairpro import FairProQwenImagePipeline

    logger.info("Loading QwenImage pipeline...")
    pipe = FairProQwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    logger.info("Enabling FairPro...")
    pipe.enable_fairpro()

    results = []
    yes_count = 0
    no_count = 0

    logger.info(f"Testing {len(prompts)} prompts...")
    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        tag = item.get("tag", "unknown")

        try:
            has_bias = pipe.check_prompt_for_bias(prompt)
            result = "yes" if has_bias else "no"
        except Exception as e:
            result = f"error: {e}"
            logger.error(f"Error checking prompt: {prompt[:50]}... - {e}")

        if result == "yes":
            yes_count += 1
        elif result == "no":
            no_count += 1

        results.append(
            {
                "prompt": prompt,
                "tag": tag,
                "bias_check": result,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(prompts)} (yes: {yes_count}, no: {no_count})"
            )

    # Print summary
    print("\n" + "=" * 70)
    print("QwenImage Bias Check Results Summary")
    print("=" * 70)
    print(f"Total prompts: {len(prompts)}")
    print(
        f"Yes (could lead to bias): {yes_count} ({yes_count / len(prompts) * 100:.1f}%)"
    )
    print(f"No (no bias concern): {no_count} ({no_count / len(prompts) * 100:.1f}%)")

    # Print by tag
    tag_results = {}
    for r in results:
        tag = r["tag"]
        if tag not in tag_results:
            tag_results[tag] = {"yes": 0, "no": 0}
        if r["bias_check"] == "yes":
            tag_results[tag]["yes"] += 1
        elif r["bias_check"] == "no":
            tag_results[tag]["no"] += 1

    print("\nBy category:")
    for tag, counts in sorted(tag_results.items()):
        total = counts["yes"] + counts["no"]
        print(
            f"  {tag}: yes={counts['yes']}, no={counts['no']} ({counts['yes'] / total * 100:.0f}% yes)"
        )

    # Save results to file
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    # Cleanup
    pipe.disable_fairpro()
    del pipe
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Test bias check on geneval prompts")
    parser.add_argument(
        "--model",
        type=str,
        choices=["sana", "qwenimage", "both"],
        default="sana",
        help="Model to test (default: sana)",
    )
    parser.add_argument(
        "--geneval-file",
        type=str,
        default="geneval/evaluation_metadata.jsonl",
        help="Path to geneval evaluation file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bias_check_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts to test (for quick testing)",
    )
    args = parser.parse_args()

    # Load prompts
    prompts = load_geneval_prompts(args.geneval_file)
    logger.info(f"Loaded {len(prompts)} prompts from {args.geneval_file}")

    if args.limit:
        prompts = prompts[: args.limit]
        logger.info(f"Limited to {len(prompts)} prompts")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    if args.model in ["sana", "both"]:
        output_file = output_dir / "sana_bias_check_results.json"
        test_sana_bias_check(prompts, str(output_file))

    if args.model in ["qwenimage", "both"]:
        output_file = output_dir / "qwenimage_bias_check_results.json"
        test_qwenimage_bias_check(prompts, str(output_file))


if __name__ == "__main__":
    main()
