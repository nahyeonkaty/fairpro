#!/usr/bin/env python3
"""FairPro: Generate fairness-aware system prompts for text-to-image generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairpro.fairpro_mixin import META_PROMPT


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_prompts(prompt_file: str) -> list[str]:
    """Load prompts from a text file.

    Args:
        prompt_file: Path to the file containing prompts.

    Returns:
        List of non-empty prompt strings.
    """
    with open(prompt_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_existing_results(output_path: Path) -> list[dict]:
    """Load existing results from JSON file if it exists.

    Args:
        output_path: Path to the output JSON file.

    Returns:
        List of existing prompt entries, or empty list if file doesn't exist.
    """
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"✓ Loaded existing file with {len(results)} prompts")
        return results
    return []


def save_results(output_path: Path, results: list[dict]) -> None:
    """Save results to JSON file.

    Args:
        output_path: Path to the output JSON file.
        results: List of prompt entries to save.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def parse_system_prompt(response: str) -> str:
    """Parse the system prompt from the model's response.

    Args:
        response: The raw model response.

    Returns:
        The extracted system prompt, or the full response if tags not found.
    """
    if "<system_prompt>" in response and "</system_prompt>" in response:
        start_idx = response.find("<system_prompt>") + len("<system_prompt>")
        end_idx = response.find("</system_prompt>")
        return response[start_idx:end_idx].strip()
    return response


def generate_system_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
) -> str:
    """Generate a fairness-aware system prompt for a given user prompt.

    Args:
        model: The language model for generation.
        tokenizer: The tokenizer for the model.
        user_prompt: The user's input prompt.

    Returns:
        The generated fairness-aware system prompt.
    """
    formatted_prompt = META_PROMPT.format(user_prompt=user_prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    return parse_system_prompt(response)


# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description="Generate fairness-aware system prompts using FairPro method"
)

parser.add_argument(
    "--prompt_file",
    type=str,
    default="data/prompts_occupations.txt",
    help="Path to the file containing prompts/occupations (default: data/prompts_occupations.txt)",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="fairpro_sp.json",
    help="Path to save the output JSON file (default: fairpro_sp.json)",
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use (default: 0)")
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Hugging Face model name for system prompt generation (default: Qwen/Qwen2.5-7B-Instruct)",
)
parser.add_argument(
    "--seeds", type=int, default=10, help="Number of seeds per prompt (default: 10)"
)
args = parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Set device and disable gradients
device = f"cuda:{args.gpu_id}"
torch.set_grad_enabled(False)

print(f"Using GPU: {device}")
print(f"Model: {args.model_name}")
print(f"# Seeds per prompt: {args.seeds}")

# Step 1: Load prompts
print_section("STEP 1: Loading prompts")
all_prompts = load_prompts(args.prompt_file)
print(f"Loaded {len(all_prompts)} prompts")
print(f"Sample prompts: {all_prompts[:5]}")

# Step 2: Load model
print_section("STEP 2: Loading model for FairPro system prompt generation")
print(f"Loading model: {args.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=torch.bfloat16, device_map={"": device}
)
print(f"Model loaded successfully on {device}!")

# Step 3: Generate FairPro system prompts
print_section("STEP 3: Generating FairPro system prompts")

output_path = Path(args.output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load existing prompts if file exists (for resuming)
all_fairpro_prompts = load_existing_results(output_path)
processed_prompts = {item["prompt"] for item in all_fairpro_prompts}

print(f"Generating FairPro system prompts for {len(all_prompts)} prompts...")
print(f"Already completed: {len(all_fairpro_prompts)} prompts")
print("=" * 80 + "\n")

# Generate system prompts for each user prompt
for i, user_prompt in enumerate(all_prompts):
    # Skip if already generated
    if user_prompt in processed_prompts:
        continue

    print(f"[{i + 1}/{len(all_prompts)}] Generating for: {user_prompt}")

    # Generate multiple seeds for each prompt
    for seed in range(args.seeds):
        print(f"Seed {seed + 1}/{args.seeds}")
        try:
            parsed_sys = generate_system_prompt(model, tokenizer, user_prompt)

            # Store the result
            all_fairpro_prompts.append(
                {"prompt": user_prompt, "parameters": {"system_prompt": parsed_sys, "seed": seed}}
            )

            # Print sample for first seed of first few prompts
            if i < 3 and seed == 0:
                print(f"System prompt: {parsed_sys[:80]}...")

        except Exception as e:
            print(f"Error generating seed {seed + 1} for '{user_prompt}': {e}")
            continue

    # Save checkpoint after each prompt (all seeds)
    save_results(output_path, all_fairpro_prompts)
    print(
        f"Completed all {args.seeds} seeds for '{user_prompt}' "
        f"(Total: {len(all_fairpro_prompts)} prompts)"
    )
    print()

# Summary
print_section("GENERATION COMPLETE!")
print(f"  Generated FairPro system prompts for {len(all_prompts)} user prompts")
print(f"  Total entries generated: {len(all_fairpro_prompts)}")
print(f"  Output saved to: {output_path}")

# Final save
save_results(output_path, all_fairpro_prompts)

print(f"\nSaved FairPro system prompts to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
print(f"Total occupations: {len(all_fairpro_prompts)}")
