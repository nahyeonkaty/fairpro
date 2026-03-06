#!/usr/bin/env python3
"""Generate images using FairProQwenImagePipeline.

This script demonstrates how to use FairProQwenImagePipeline to generate
images with fairness-aware system prompts.

Usage:
    # Generate a single image with FairPro
    python generate_fairpro.py --prompt "A doctor examining a patient"

    # Generate comparison images (with and without FairPro)
    python generate_fairpro.py --prompt "A CEO giving a presentation" --compare

    # Batch generation from file
    python generate_fairpro.py --prompt-file data/prompts_simple.txt --output-dir outputs/

    # Custom model and settings
    python generate_fairpro.py --prompt "An engineer at work" \
        --model "Qwen/Qwen-Image" \
        --steps 30 --seed 42

Note: By default, FairPro uses QwenImage's built-in Qwen2.5 text encoder.
You can override this with --fairpro-model (and optional quantization).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fairpro import FairProQwenImagePipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images using FairProQwenImagePipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for image generation",
    )
    input_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to file containing prompts (one per line)",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-Image",
        help="QwenImage model to use (default: Qwen/Qwen-Image)",
    )
    parser.add_argument(
        "--fairpro-model",
        type=str,
        default=None,
        help=(
            "Optional separate LLM for FairPro prompt generation. "
            "Default: use QwenImage built-in text encoder."
        ),
    )
    parser.add_argument(
        "--fairpro-quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Optional quantization mode for external FairPro LLM",
    )

    # Generation options
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
        help="True CFG scale (default: 4.0)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, worst quality, blurry, ugly, distorted",
        help="Negative prompt for guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # FairPro options
    parser.add_argument(
        "--no-fairpro",
        action="store_true",
        help="Disable FairPro (use default system prompt)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison images (with and without FairPro)",
    )
    parser.add_argument(
        "--fairpro-batch-size",
        type=int,
        default=8,
        help="Batch size for FairPro system prompt generation (default: 8)",
    )
    parser.add_argument(
        "--fairpro-no-cache",
        action="store_true",
        help="Disable in-memory FairPro cache",
    )
    parser.add_argument(
        "--fairpro-num-candidates",
        type=int,
        default=1,
        help="Number of sampled FairPro candidates per prompt (default: 1)",
    )
    parser.add_argument(
        "--fairpro-select-best",
        action="store_true",
        help="Score and select best candidate using fairness+faithfulness",
    )
    parser.add_argument(
        "--fairpro-fairness-weight",
        type=float,
        default=0.6,
        help="Fairness weight for candidate selector (default: 0.6)",
    )
    parser.add_argument(
        "--fairpro-faithfulness-weight",
        type=float,
        default=0.4,
        help="Faithfulness weight for candidate selector (default: 0.4)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated images (default: outputs)",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save the generated system prompt to a text file",
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for image generation (default: cuda:0)",
    )

    return parser.parse_args()


def load_prompts(prompt_file: str) -> list[str]:
    """Load prompts from a text file.

    Args:
        prompt_file: Path to file containing prompts.

    Returns:
        List of prompts.
    """
    with open(prompt_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def create_output_filename(
    prompt: str,
    output_dir: Path,
    suffix: str = "",
    seed: int | None = None,
) -> Path:
    """Create a unique output filename.

    Args:
        prompt: The generation prompt.
        output_dir: Output directory.
        suffix: Optional suffix for the filename.
        seed: Optional seed value.

    Returns:
        Path to the output file.
    """
    # Create a short version of the prompt for the filename
    short_prompt = prompt[:50].replace(" ", "_").replace("/", "-")
    short_prompt = "".join(c for c in short_prompt if c.isalnum() or c in "_-")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if seed is not None:
        filename = f"{short_prompt}_{suffix}_seed{seed}_{timestamp}.png"
    else:
        filename = f"{short_prompt}_{suffix}_{timestamp}.png"

    return output_dir / filename


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FairPro Image Generation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output Directory: {output_dir}")
    if args.fairpro_model:
        print(f"FairPro LLM: {args.fairpro_model}")
        print(f"FairPro Quantization: {args.fairpro_quantization or 'none'}")
    else:
        print("FairPro LLM: uses built-in Qwen2.5 text encoder")
    print("=" * 70)

    # Load the pipeline
    print("\n[1/3] Loading QwenImage pipeline...")
    pipe = FairProQwenImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    # Enable FairPro if not disabled (uses built-in text encoder)
    if not args.no_fairpro:
        print("\n[2/3] Enabling FairPro...")
        pipe.enable_fairpro(
            model_name=args.fairpro_model,
            quantization=args.fairpro_quantization,
        )
    else:
        print("\n[2/3] FairPro disabled, using default system prompt")

    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = load_prompts(args.prompt_file)
        print(f"\nLoaded {len(prompts)} prompts from {args.prompt_file}")

    # Generation settings
    gen_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
    }
    fairpro_kwargs = {
        "fairpro_batch_size": args.fairpro_batch_size,
        "fairpro_use_cache": not args.fairpro_no_cache,
        "fairpro_num_candidates": args.fairpro_num_candidates,
        "fairpro_select_best": args.fairpro_select_best,
        "fairpro_fairness_weight": args.fairpro_fairness_weight,
        "fairpro_faithfulness_weight": args.fairpro_faithfulness_weight,
    }

    print("\n[3/3] Generating images...")
    print(f"  - Size: {args.width}x{args.height}")
    print(f"  - Steps: {args.steps}")
    print(f"  - Guidance Scale: {args.guidance_scale}")
    print(f"  - True CFG Scale: {args.true_cfg_scale}")
    if args.seed is not None:
        print(f"  - Seed: {args.seed}")
    if not args.no_fairpro:
        print(f"  - FairPro Batch Size: {args.fairpro_batch_size}")
        print(f"  - FairPro Cache: {not args.fairpro_no_cache}")
        print(f"  - FairPro Candidates: {args.fairpro_num_candidates}")
        print(f"  - FairPro Selector: {args.fairpro_select_best}")
    print()

    # Generate images
    for i, prompt in enumerate(prompts, 1):
        print("-" * 70)
        print(f"Prompt {i}/{len(prompts)}: {prompt}")
        print("-" * 70)

        # Set up generator for reproducibility
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)

        if args.compare and not args.no_fairpro:
            # Generate comparison images
            print("\nGenerating comparison images...")
            default_generator = None
            fairpro_generator = None
            if args.seed is not None:
                default_generator = torch.Generator(device="cpu").manual_seed(args.seed)
                fairpro_generator = torch.Generator(device="cpu").manual_seed(args.seed)

            default_image = pipe(
                prompt=prompt,
                generator=default_generator,
                use_fairpro=False,
                **gen_kwargs,
            ).images[0]

            system_prompt = pipe.generate_fairpro_system_prompt(
                prompt,
                batch_size=args.fairpro_batch_size,
                use_cache=not args.fairpro_no_cache,
                num_candidates=args.fairpro_num_candidates,
                select_best=args.fairpro_select_best,
                fairness_weight=args.fairpro_fairness_weight,
                faithfulness_weight=args.fairpro_faithfulness_weight,
            )
            fairpro_image = pipe(
                prompt=prompt,
                generator=fairpro_generator,
                use_fairpro=True,
                fairpro_system_prompts=system_prompt,
                **gen_kwargs,
                **fairpro_kwargs,
            ).images[0]

            # Save default image
            default_path = create_output_filename(
                prompt, output_dir, "default", args.seed
            )
            default_image.save(default_path)
            print(f"  ✓ Default image saved: {default_path}")

            # Save FairPro image
            fairpro_path = create_output_filename(
                prompt, output_dir, "fairpro", args.seed
            )
            fairpro_image.save(fairpro_path)
            print(f"  ✓ FairPro image saved: {fairpro_path}")

            # Optionally save system prompt
            if args.save_prompt:
                prompt_path = fairpro_path.with_suffix(".txt")
                with open(prompt_path, "w") as f:
                    f.write(f"User Prompt: {prompt}\n\n")
                    f.write(f"FairPro System Prompt:\n{system_prompt}\n")
                print(f"  ✓ System prompt saved: {prompt_path}")

            print(f"\nGenerated FairPro System Prompt:\n{system_prompt}\n")

        else:
            # Generate single image
            use_fairpro = not args.no_fairpro and pipe.fairpro_enabled

            result = pipe(
                prompt=prompt,
                generator=generator,
                use_fairpro=use_fairpro,
                **fairpro_kwargs,
                **gen_kwargs,
            )

            image = result.images[0]

            # Save image
            suffix = "fairpro" if use_fairpro else "default"
            image_path = create_output_filename(prompt, output_dir, suffix, args.seed)
            image.save(image_path)
            print(f"  ✓ Image saved: {image_path}")

        # Clear CUDA cache between prompts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
