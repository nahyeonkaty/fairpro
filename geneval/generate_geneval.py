#!/usr/bin/env python3
"""Generate images for GenEval prompts using QwenImage and SANA with and without FairPro.

This script generates images using:
- QwenImage (without FairPro)
- QwenImage + FairPro
- SANA (without FairPro)
- SANA + FairPro

Each prompt is generated with 4 seeds (0, 1, 2, 3).

Usage:
    python generate_geneval.py --model qwenimage --output-dir outputs/geneval
    python generate_geneval.py --model qwenimage --fairpro --output-dir outputs/geneval
    python generate_geneval.py --model sana --output-dir outputs/geneval
    python generate_geneval.py --model sana --fairpro --output-dir outputs/geneval

    # Run all configurations
    python generate_geneval.py --all --output-dir outputs/geneval
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

GENEVAL_METADATA_URL = (
    "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images for GenEval prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--metadata-file",
        type=str,
        default="geneval/evaluation_metadata.jsonl",
        help="Path to GenEval evaluation_metadata.jsonl (default: data/evaluation_metadata.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwenimage", "sana"],
        help="Model to use for generation",
    )
    parser.add_argument(
        "--fairpro",
        action="store_true",
        help="Enable FairPro for fairness-aware generation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 4 configurations (QwenImage, QwenImage+FairPro, SANA, SANA+FairPro)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Seeds to use for generation (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/geneval",
        help="Output directory for generated images (default: outputs/geneval)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0)",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale for QwenImage (default: 4.0)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, worst quality, blurry, ugly, distorted",
        help="Negative prompt for guidance",
    )
    parser.add_argument(
        "--qwenimage-model",
        type=str,
        default="Qwen/Qwen-Image",
        help="QwenImage model name (default: Qwen/Qwen-Image)",
    )
    parser.add_argument(
        "--sana-model",
        type=str,
        default="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        help="SANA model name (default: Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for prompts (default: 0)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for prompts (default: None, process all)",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip saving grid images",
    )

    args = parser.parse_args()

    if not args.all and not args.model:
        parser.error("Either --model or --all must be specified")

    return args


def load_metadata(metadata_file: str) -> list[dict]:
    """Load GenEval metadata from evaluation_metadata.jsonl.

    If the file doesn't exist, downloads it from GitHub.
    Returns list of metadata dicts containing 'prompt', 'tag', 'include', etc.
    """
    metadata_path = Path(metadata_file)

    # Download if file doesn't exist
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_file}")
        print(f"Downloading from {GENEVAL_METADATA_URL}...")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        import urllib.request

        urllib.request.urlretrieve(GENEVAL_METADATA_URL, metadata_path)
        print(f"Downloaded to: {metadata_path}")

    print(f"Loading metadata from: {metadata_file}")
    metadata_list = []
    with open(metadata_file) as f:
        for line in f:
            line = line.strip()
            if line:
                metadata_list.append(json.loads(line))
    return metadata_list


def save_grid(images: list, output_path: Path, n_rows: int = 4) -> None:
    """Save a grid of images."""
    tensors = [ToTensor()(img) for img in images]
    grid = make_grid(torch.stack(tensors), nrow=n_rows, padding=0)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    grid = Image.fromarray(grid.astype(np.uint8))
    grid.save(output_path)


def generate_with_qwenimage(
    prompts: list[str],
    seeds: list[int],
    output_dir: Path,
    use_fairpro: bool,
    args: argparse.Namespace,
    start_idx: int = 0,
    metadata_list: list[dict] | None = None,
) -> None:
    """Generate images using QwenImage pipeline.

    Output format matches diffusers_generate.py:
    - {output_dir}/{index:05d}/samples/{seed:05d}.png
    - {output_dir}/{index:05d}/metadata.jsonl
    - {output_dir}/{index:05d}/grid.png (optional)
    """
    from fairpro import FairProQwenImagePipeline

    config_name = "qwenimage_fairpro" if use_fairpro else "qwenimage"
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Configuration: {config_name.upper()}")
    print(f"Output Directory: {config_dir}")
    print("=" * 80)

    # Load pipeline
    print("\nLoading QwenImage pipeline...")
    pipe = FairProQwenImagePipeline.from_pretrained(
        args.qwenimage_model,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    if use_fairpro:
        print("Enabling FairPro...")
        pipe.enable_fairpro()

    gen_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
    }

    total = len(prompts)

    for i, prompt in enumerate(prompts):
        # Use global index (start_idx + i) for folder naming to match original format
        global_idx = start_idx + i
        prompt_dir = config_dir / f"{global_idx:05d}"
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata (use full GenEval metadata if available)
        if metadata_list is not None and global_idx < len(metadata_list):
            metadata = metadata_list[global_idx]
        else:
            # Fallback: create minimal metadata with default tag
            metadata = {"prompt": prompt, "tag": "single_object"}
        with open(prompt_dir / "metadata.jsonl", "w") as f:
            json.dump(metadata, f)

        system_prompt_log_path = prompt_dir / "system_prompts.jsonl"

        print(f"\nPrompt ({i + 1:>3}/{total}): '{prompt}'")

        all_images = []
        for seed_idx, seed in enumerate(seeds):
            print(f"  Generating seed {seed}...")

            # Determine system prompt for this prompt+seed (save in single log file)
            if use_fairpro:
                system_prompt = pipe.generate_fairpro_system_prompt(prompt)
            else:
                system_prompt = pipe.get_default_system_prompt()

            with open(system_prompt_log_path, "a") as f:
                json.dump(
                    {
                        "index": global_idx,
                        "seed": seed,
                        "seed_idx": seed_idx,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                    },
                    f,
                )
                f.write("\n")

            generator = torch.Generator(device="cpu").manual_seed(seed)

            result = pipe(
                prompt=prompt,
                generator=generator,
                use_fairpro=use_fairpro,
                fairpro_system_prompts=system_prompt if use_fairpro else None,
                **gen_kwargs,
            )

            image = result.images[0]
            image_path = sample_dir / f"{seed_idx:05d}.png"
            image.save(image_path)
            print(f"    ✓ Saved: {image_path}")

            all_images.append(image)

            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save grid
        if not args.skip_grid:
            grid_path = prompt_dir / "grid.png"
            save_grid(all_images, grid_path, n_rows=len(seeds))
            print(f"  ✓ Grid saved: {grid_path}")

    # Cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✓ {config_name} generation complete!")


def generate_with_sana(
    prompts: list[str],
    seeds: list[int],
    output_dir: Path,
    use_fairpro: bool,
    args: argparse.Namespace,
    start_idx: int = 0,
    metadata_list: list[dict] | None = None,
) -> None:
    """Generate images using SANA pipeline.

    Output format matches diffusers_generate.py:
    - {output_dir}/{index:05d}/samples/{seed:05d}.png
    - {output_dir}/{index:05d}/metadata.jsonl
    - {output_dir}/{index:05d}/grid.png (optional)
    """
    from fairpro import FairProSanaPipeline

    config_name = "sana_fairpro" if use_fairpro else "sana"
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)


    print("=" * 80)
    print(f"Configuration: {config_name.upper()}")
    print(f"Output Directory: {config_dir}")
    print("=" * 80)

    # Load pipeline
    print("\nLoading SANA pipeline...")
    pipe = FairProSanaPipeline.from_pretrained(
        args.sana_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if use_fairpro:
        print("Enabling FairPro...")
        pipe.enable_fairpro()

    gen_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
    }

    total = len(prompts)

    for i, prompt in enumerate(prompts):
        # Use global index (start_idx + i) for folder naming to match original format
        global_idx = start_idx + i
        prompt_dir = config_dir / f"{global_idx:05d}"
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata (use full GenEval metadata if available)
        if metadata_list is not None and global_idx < len(metadata_list):
            metadata = metadata_list[global_idx]
        else:
            # Fallback: create minimal metadata with default tag
            metadata = {"prompt": prompt, "tag": "single_object"}
        with open(prompt_dir / "metadata.jsonl", "w") as f:
            json.dump(metadata, f)

        system_prompt_log_path = prompt_dir / "system_prompts.jsonl"

        print(f"\nPrompt ({i + 1:>3}/{total}): '{prompt}'")

        all_images = []
        for seed_idx, seed in enumerate(seeds):
            print(f"  Generating seed {seed}...")

            # Determine system prompt for this prompt+seed (save in single log file)
            if use_fairpro:
                system_prompt = pipe.generate_fairpro_system_prompt(prompt)
            else:
                system_prompt = pipe.get_default_system_prompt()

            with open(system_prompt_log_path, "a") as f:
                json.dump(
                    {
                        "index": global_idx,
                        "seed": seed,
                        "seed_idx": seed_idx,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                    },
                    f,
                )
                f.write("\n")

            generator = torch.Generator(device="cuda").manual_seed(seed)

            result = pipe(
                prompt=prompt,
                generator=generator,
                use_fairpro=use_fairpro,
                fairpro_system_prompts=system_prompt if use_fairpro else None,
                **gen_kwargs,
            )

            image = result.images[0]
            image_path = sample_dir / f"{seed_idx:05d}.png"
            image.save(image_path)
            print(f"    ✓ Saved: {image_path}")

            all_images.append(image)

            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save grid
        if not args.skip_grid:
            grid_path = prompt_dir / "grid.png"
            save_grid(all_images, grid_path, n_rows=len(seeds))
            print(f"  ✓ Grid saved: {grid_path}")

    # Cleanup
    if use_fairpro:
        pipe.disable_fairpro()
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✓ {config_name} generation complete!")


def main():
    """Main entry point."""
    args = parse_args()

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent))

    # Load GenEval metadata (downloads if not exists)
    metadata_list = load_metadata(args.metadata_file)

    # Extract prompts from metadata
    prompts = [m["prompt"] for m in metadata_list]

    # Apply start/end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(prompts)
    prompts = prompts[start_idx:end_idx]

    print(f"Loaded {len(prompts)} prompts from {args.metadata_file}")
    print(f"  (indices {start_idx} to {end_idx})")
    print(f"Seeds: {args.seeds}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Run all 4 configurations
        configurations = [
            ("qwenimage", False),
            ("qwenimage", True),
            ("sana", False),
            ("sana", True),
        ]

        for model, use_fairpro in configurations:
            if model == "qwenimage":
                generate_with_qwenimage(
                    prompts, args.seeds, output_dir, use_fairpro, args, start_idx, metadata_list
                )
            else:
                generate_with_sana(prompts, args.seeds, output_dir, use_fairpro, args, start_idx, metadata_list)

    else:
        # Run single configuration
        if args.model == "qwenimage":
            generate_with_qwenimage(prompts, args.seeds, output_dir, args.fairpro, args, start_idx, metadata_list)
        else:
            generate_with_sana(prompts, args.seeds, output_dir, args.fairpro, args, start_idx, metadata_list)

    print("\n" + "=" * 80)
    print("All generations complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
