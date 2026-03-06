#!/usr/bin/env python3
"""Benchmark script to measure additional computational time of FairPro.

This script measures:
1. System prompt generation time
2. Image generation time (with and without FairPro)
3. Total pipeline time comparison

Usage:
    python benchmark_fairpro.py --model sana
    python benchmark_fairpro.py --model qwenimage
    python benchmark_fairpro.py --model sana --num-runs 5
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model: str
    prompt: str
    num_runs: int
    # System prompt generation
    avg_system_prompt_time: float
    min_system_prompt_time: float
    max_system_prompt_time: float
    # Image generation without FairPro
    avg_gen_time_baseline: float
    min_gen_time_baseline: float
    max_gen_time_baseline: float
    # Image generation with FairPro (excluding system prompt generation)
    avg_gen_time_fairpro: float
    min_gen_time_fairpro: float
    max_gen_time_fairpro: float
    # Total time with FairPro (including system prompt generation)
    avg_total_time_fairpro: float
    # Overhead
    overhead_percent: float

    def __str__(self) -> str:
        return f"""
{"=" * 70}
FairPro Benchmark Results - {self.model.upper()}
{"=" * 70}
Prompt: "{self.prompt}"
Number of runs: {self.num_runs}

System Prompt Generation:
  Average: {self.avg_system_prompt_time:.3f}s
  Min:     {self.min_system_prompt_time:.3f}s
  Max:     {self.max_system_prompt_time:.3f}s

Image Generation (Baseline - no FairPro):
  Average: {self.avg_gen_time_baseline:.3f}s
  Min:     {self.min_gen_time_baseline:.3f}s
  Max:     {self.max_gen_time_baseline:.3f}s

Image Generation (FairPro - excluding prompt gen):
  Average: {self.avg_gen_time_fairpro:.3f}s
  Min:     {self.min_gen_time_fairpro:.3f}s
  Max:     {self.max_gen_time_fairpro:.3f}s

Total Time with FairPro (including prompt gen):
  Average: {self.avg_total_time_fairpro:.3f}s

FairPro Overhead:
  System prompt generation adds: {self.avg_system_prompt_time:.3f}s ({self.overhead_percent:.1f}%)
{"=" * 70}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FairPro computational overhead"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sana", "qwenimage"],
        default="sana",
        help="Model to benchmark (default: sana)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for averaging (default: 3)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A doctor examining a patient in a hospital room",
        help="Prompt to use for benchmarking",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save benchmark images (optional)",
    )
    parser.add_argument(
        "--sana-model",
        type=str,
        default="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        help="Sana model name",
    )
    parser.add_argument(
        "--qwenimage-model",
        type=str,
        default="Qwen/Qwen-Image",
        help="QwenImage model name",
    )
    return parser.parse_args()


def benchmark_sana(args: argparse.Namespace) -> BenchmarkResult:
    """Benchmark FairPro overhead for Sana pipeline."""
    from fairpro import FairProSanaPipeline

    print(f"Loading Sana pipeline: {args.sana_model}")
    pipe = FairProSanaPipeline.from_pretrained(
        args.sana_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print("Enabling FairPro...")
    pipe.enable_fairpro()

    gen_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": 4.5,
    }

    # Warmup runs
    print(f"\nRunning {args.warmup} warmup run(s)...")
    for _ in range(args.warmup):
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        _ = pipe(
            prompt=args.prompt, generator=generator, use_fairpro=False, **gen_kwargs
        )
        torch.cuda.empty_cache()

    # Benchmark system prompt generation
    print(f"\nBenchmarking system prompt generation ({args.num_runs} runs)...")
    system_prompt_times = []
    generated_prompts = []
    for i in range(args.num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        sp = pipe.generate_fairpro_system_prompt(args.prompt)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        system_prompt_times.append(elapsed)
        generated_prompts.append(sp)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    # Benchmark baseline generation (no FairPro)
    print(f"\nBenchmarking baseline generation ({args.num_runs} runs)...")
    baseline_times = []
    for i in range(args.num_runs):
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipe(
            prompt=args.prompt, generator=generator, use_fairpro=False, **gen_kwargs
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        baseline_times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
        if args.output_dir and i == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            result.images[0].save(Path(args.output_dir) / "sana_baseline.png")
        torch.cuda.empty_cache()

    # Benchmark FairPro generation (using pre-generated system prompt)
    print(f"\nBenchmarking FairPro generation ({args.num_runs} runs)...")
    fairpro_times = []
    for i in range(args.num_runs):
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipe(
            prompt=args.prompt,
            generator=generator,
            use_fairpro=True,
            fairpro_system_prompts=generated_prompts[i],
            **gen_kwargs,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fairpro_times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
        if args.output_dir and i == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            result.images[0].save(Path(args.output_dir) / "sana_fairpro.png")
        torch.cuda.empty_cache()

    # Cleanup
    pipe.disable_fairpro()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate results
    avg_sp_time = sum(system_prompt_times) / len(system_prompt_times)
    avg_baseline = sum(baseline_times) / len(baseline_times)
    avg_fairpro = sum(fairpro_times) / len(fairpro_times)
    overhead_percent = (avg_sp_time / avg_baseline) * 100

    return BenchmarkResult(
        model="sana",
        prompt=args.prompt,
        num_runs=args.num_runs,
        avg_system_prompt_time=avg_sp_time,
        min_system_prompt_time=min(system_prompt_times),
        max_system_prompt_time=max(system_prompt_times),
        avg_gen_time_baseline=avg_baseline,
        min_gen_time_baseline=min(baseline_times),
        max_gen_time_baseline=max(baseline_times),
        avg_gen_time_fairpro=avg_fairpro,
        min_gen_time_fairpro=min(fairpro_times),
        max_gen_time_fairpro=max(fairpro_times),
        avg_total_time_fairpro=avg_sp_time + avg_fairpro,
        overhead_percent=overhead_percent,
    )


def benchmark_qwenimage(args: argparse.Namespace) -> BenchmarkResult:
    """Benchmark FairPro overhead for QwenImage pipeline."""
    from fairpro import FairProQwenImagePipeline

    print(f"Loading QwenImage pipeline: {args.qwenimage_model}")
    pipe = FairProQwenImagePipeline.from_pretrained(
        args.qwenimage_model,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    print("Enabling FairPro...")
    pipe.enable_fairpro()

    gen_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": 5.0,
        "true_cfg_scale": 4.0,
        "negative_prompt": "low quality, worst quality, blurry, ugly",
    }

    # Warmup runs
    print(f"\nRunning {args.warmup} warmup run(s)...")
    for _ in range(args.warmup):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        _ = pipe(
            prompt=args.prompt, generator=generator, use_fairpro=False, **gen_kwargs
        )
        torch.cuda.empty_cache()

    # Benchmark system prompt generation
    print(f"\nBenchmarking system prompt generation ({args.num_runs} runs)...")
    system_prompt_times = []
    generated_prompts = []
    for i in range(args.num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        sp = pipe.generate_fairpro_system_prompt(args.prompt)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        system_prompt_times.append(elapsed)
        generated_prompts.append(sp)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    # Benchmark baseline generation (no FairPro)
    print(f"\nBenchmarking baseline generation ({args.num_runs} runs)...")
    baseline_times = []
    for i in range(args.num_runs):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipe(
            prompt=args.prompt, generator=generator, use_fairpro=False, **gen_kwargs
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        baseline_times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
        if args.output_dir and i == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            result.images[0].save(Path(args.output_dir) / "qwenimage_baseline.png")
        torch.cuda.empty_cache()

    # Benchmark FairPro generation (using pre-generated system prompt)
    print(f"\nBenchmarking FairPro generation ({args.num_runs} runs)...")
    fairpro_times = []
    for i in range(args.num_runs):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipe(
            prompt=args.prompt,
            generator=generator,
            use_fairpro=True,
            fairpro_system_prompts=generated_prompts[i],
            **gen_kwargs,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fairpro_times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
        if args.output_dir and i == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            result.images[0].save(Path(args.output_dir) / "qwenimage_fairpro.png")
        torch.cuda.empty_cache()

    # Cleanup
    pipe.disable_fairpro()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate results
    avg_sp_time = sum(system_prompt_times) / len(system_prompt_times)
    avg_baseline = sum(baseline_times) / len(baseline_times)
    avg_fairpro = sum(fairpro_times) / len(fairpro_times)
    overhead_percent = (avg_sp_time / avg_baseline) * 100

    return BenchmarkResult(
        model="qwenimage",
        prompt=args.prompt,
        num_runs=args.num_runs,
        avg_system_prompt_time=avg_sp_time,
        min_system_prompt_time=min(system_prompt_times),
        max_system_prompt_time=max(system_prompt_times),
        avg_gen_time_baseline=avg_baseline,
        min_gen_time_baseline=min(baseline_times),
        max_gen_time_baseline=max(baseline_times),
        avg_gen_time_fairpro=avg_fairpro,
        min_gen_time_fairpro=min(fairpro_times),
        max_gen_time_fairpro=max(fairpro_times),
        avg_total_time_fairpro=avg_sp_time + avg_fairpro,
        overhead_percent=overhead_percent,
    )


def main():
    args = parse_args()

    print("=" * 70)
    print("FairPro Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Inference steps: {args.steps}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Warmup runs: {args.warmup}")
    print("=" * 70)

    if args.model == "sana":
        result = benchmark_sana(args)
    else:
        result = benchmark_qwenimage(args)

    print(result)

    # Save results to file
    if args.output_dir:
        output_path = Path(args.output_dir) / f"benchmark_{args.model}.txt"
        with open(output_path, "w") as f:
            f.write(str(result))
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
