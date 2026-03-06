#!/usr/bin/env python3
"""FairPro Image Generation: Generate comparison images using default vs FairPro system prompts.

This script takes the JSON output from fairpro.py and generates images for comparison.
"""

from __future__ import annotations

import argparse

# Parse arguments FIRST before any imports that use CUDA
parser = argparse.ArgumentParser(
    description="Generate comparison images using FairPro system prompts"
)

parser.add_argument(
    "--input_json",
    type=str,
    default="fairpro_sp.json",
    help="Path to the JSON file containing FairPro system prompts (default: fairpro_sp.json)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Base output directory for generated images (default: output)",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen-Image",
    help="Hugging Face model name for T2I generation (default: Qwen/Qwen-Image)",
)
parser.add_argument(
    "--gpu_ids",
    type=int,
    nargs=2,
    default=[0, 1],
    help="Two GPU IDs to use for model distribution (default: 0 1)",
)
parser.add_argument(
    "--height", type=int, default=1024, help="Image height (default: 1024)"
)
parser.add_argument(
    "--width", type=int, default=1024, help="Image width (default: 1024)"
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=20,
    help="Number of inference steps (default: 20)",
)
parser.add_argument(
    "--true_cfg_scale", type=float, default=4.0, help="True CFG scale (default: 4.0)"
)
parser.add_argument(
    "--negative_prompt",
    type=str,
    default="low quality, worst quality, blurry, ugly",
    help='Negative prompt for generation (default: "low quality, worst quality, blurry, ugly")',
)

args = parser.parse_args()

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch or any CUDA libraries
import os  # noqa: E402

gpu0, gpu1 = args.gpu_ids
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu0},{gpu1}"

# Now import torch and other libraries
import gc  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from diffusers import QwenImagePipeline  # noqa: E402

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n{{}}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_fairpro_prompts(input_path: Path) -> dict[str, list[dict]]:
    """Load and group FairPro system prompts from JSON file.

    Args:
        input_path: Path to the input JSON file.

    Returns:
        Dictionary mapping prompts to their iterations (system_prompt, seed).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    with open(input_path) as f:
        fairpro_data = json.load(f)

    print(f"Loaded {len(fairpro_data)} entries from {input_path}")

    # Group by prompt to get all iterations
    prompt_groups: dict[str, list[dict]] = {}
    for entry in fairpro_data:
        prompt = entry["prompt"]
        system_prompt = entry["parameters"]["system_prompt"]
        seed = entry["parameters"]["seed"]

        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append({"system_prompt": system_prompt, "seed": seed})

    print(f"Found {len(prompt_groups)} unique prompts")
    print(f"Sample prompts: {list(prompt_groups.keys())[:3]}")

    return prompt_groups


def generate_image(
    pipeline: QwenImagePipeline,
    prompt: str,
    system_prompt: str,
    seed: int,
    output_path: Path,
    is_default: bool = False,
) -> bool:
    """Generate and save a single image with the given system prompt.

    Args:
        pipeline: The T2I pipeline to use.
        prompt: The user prompt for image generation.
        system_prompt: The system prompt to use.
        seed: Random seed for generation.
        output_path: Path to save the generated image.
        is_default: Whether this is using the default system prompt (for logging).

    Returns:
        True if image was generated, False if skipped (already exists).
    """
    label = "DEFAULT" if is_default else "FAIRPRO"
    step = "1/2" if is_default else "2/2"

    if output_path.exists():
        print(f"[{step}] {label} image already exists, skipping...")
        return False

    print(f"[{step}] Generating with {label} system prompt...")

    # Set prompt template
    pipeline.prompt_template_encode = PROMPT_TEMPLATE.format(
        system_prompt=system_prompt
    )
    if is_default:
        pipeline.prompt_template_encode_start_idx = 34
    else:
        pipeline.prompt_template_encode_start_idx = (
            len(pipeline.tokenizer.encode(pipeline.prompt_template_encode)) - 6
        )

    # Generate image
    gen_params = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
    }
    result = pipeline(**gen_params)
    image = result.images[0]

    # Save image
    image.save(output_path)
    print(f"✓ Saved to: {output_path}")

    # Clear memory
    del result, image
    torch.cuda.empty_cache()

    return True


def clear_memory() -> None:
    """Clear GPU memory and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()


def create_safe_filename(prompt: str, seed: int, max_length: int = 50) -> str:
    """Create a filesystem-safe filename from a prompt.

    Args:
        prompt: The original prompt string.
        seed: The seed number to append.
        max_length: Maximum length for the prompt portion.

    Returns:
        A safe filename string.
    """
    safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:max_length]
    return f"{safe_prompt}_{seed}.png"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Disable gradients
torch.set_grad_enabled(False)

print(f"Using only GPUs: {gpu0} and {gpu1} (mapped to cuda:0 and cuda:1)")
print(f"Available devices: {torch.cuda.device_count()}")
print(f"T2I Model: {args.model_name}")

# Step 1: Load FairPro system prompts
print_section("STEP 1: Loading FairPro system prompts")
input_json_path = Path(args.input_json)
prompt_groups = load_fairpro_prompts(input_json_path)

# Step 2: Load T2I pipeline
print_section("STEP 2: Loading T2I pipeline")
print(f"Loading Qwen-Image pipeline on GPU {gpu0} and {gpu1}...")

pipeline = QwenImagePipeline.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
pipeline.set_progress_bar_config(disable=False)

print(f"T2I Pipeline loaded successfully on GPU {gpu0} and {gpu1}!")
clear_memory()

# Step 3: Generate images
print_section("STEP 3: Generating images")

# Create output directories
base_output_dir = Path(args.output_dir)
default_dir = base_output_dir / "default"
fairpro_dir = base_output_dir / "fairpro"

default_dir.mkdir(parents=True, exist_ok=True)
fairpro_dir.mkdir(parents=True, exist_ok=True)

print("Output directories:")
print(f"  Default: {default_dir}")
print(f"  FairPro: {fairpro_dir}")
print("=" * 80 + "\n")

successful_generations = 0
total_images = sum(len(iterations) for iterations in prompt_groups.values())
image_count = 0

for prompt_idx, (prompt, iterations) in enumerate(prompt_groups.items()):
    print(f"\n[{prompt_idx + 1}/{len(prompt_groups)}] Processing prompt: '{prompt}'")
    print(f"Generating {len(iterations)} iterations")
    print("-" * 80)

    for iter_data in iterations:
        image_count += 1
        system_prompt = iter_data["system_prompt"]
        seed = iter_data["seed"]

        print(f"[{image_count}/{total_images}] Iteration {seed}")

        filename = create_safe_filename(prompt, seed)
        default_path = default_dir / filename
        fairpro_path = fairpro_dir / filename

        # Skip if both images already exist
        if default_path.exists() and fairpro_path.exists():
            print("Both images already exist, skipping...")
            successful_generations += 1
            continue

        # Generate with DEFAULT system prompt
        generate_image(
            pipeline, prompt, DEFAULT_SYSTEM_PROMPT, seed, default_path, is_default=True
        )

        # Generate with FAIRPRO system prompt
        generate_image(
            pipeline, prompt, system_prompt, seed, fairpro_path, is_default=False
        )

        successful_generations += 1
        clear_memory()

# Summary
print_section("GENERATION COMPLETE!")
print(f"  Successfully generated {successful_generations}/{total_images} image pairs")
print(f"  Default images saved in: {default_dir}")
print(f"  FairPro images saved in: {fairpro_dir}")
print(f"  Input JSON: {input_json_path}")
print("=" * 80)
