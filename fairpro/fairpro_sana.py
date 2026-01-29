"""FairPro Sana Pipeline with fairness-aware system prompt generation."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from diffusers import SanaPipeline
from diffusers.pipelines.sana.pipeline_sana import SanaPipelineOutput

from fairpro.fairpro_mixin import FairProMixin

logger = logging.getLogger(__name__)

# Sana default system prompt (matches diffusers SanaPipeline complex_human_instruction)
SANA_DEFAULT_SYSTEM_PROMPT = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]

# Sana-specific meta prompt template
SANA_META_PROMPT_TEMPLATE = """You are an intelligent, unbiased assistant.
Your goal is to design a fair instruction that guides detailed, accurate, and globally inclusive image descriptions.

Consider the following user prompt:
"{{user_prompt}}"

Identify what would be the problem if the original system prompt were used for the above user prompt. Focus on:
- Potential social stereotypes or biases that might emerge
- Lack of diversity or inclusivity
- Cultural or demographic assumptions

Think of a concise analysis of the potential issues.
Based on this analysis, create a system prompt that addresses these issues and promotes fairness, diversity, and inclusivity.

CAUTION: If the user prompt does not involve humans (e.g., inanimate objects, landscapes, animals), you MUST output the 'Original system prompt' exactly as written below, word-for-word.
Do not add any extra clauses about fairness, diversity, or inclusivity to this string.
Original system instruction: "{SANA_DEFAULT_SYSTEM_PROMPT}"

Output constraint: Write only the final revised system instruction with no commentary, explanations, or reasoning.
The last line must exactly be “User Prompt: ”.
"""


class FairProSanaPipeline(FairProMixin, SanaPipeline):
    """Sana Pipeline with FairPro fairness-aware system prompt generation.

    This pipeline extends SanaPipeline to automatically generate and apply
    fairness-aware system prompts for each user prompt before image generation.

    Key features:
    - Uses Sana's built-in Gemma text encoder for system prompt generation by default
    - Supports per-prompt system prompts (different system prompt for each prompt)
    - Batched system prompt generation for efficiency
    - API compatible with original SanaPipeline

    Example:
        ```python
        from pipelines import FairProSanaPipeline

        # Load the pipeline
        pipe = FairProSanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        # Enable FairPro functionality (uses built-in Gemma text encoder)
        pipe.enable_fairpro()

        # Single prompt generation with FairPro
        image = pipe(
            prompt="A doctor examining a patient",
            use_fairpro=True,
        ).images[0]

        # Multiple prompts with per-prompt FairPro system prompts
        images = pipe(
            prompt=["A doctor", "An engineer", "A teacher"],
            use_fairpro=True,
        ).images
        ```
    """

    # Default FairPro model - Gemma 2 2B Instruct (same architecture as Sana's text encoder)
    DEFAULT_FAIRPRO_MODEL = "google/gemma-2-2b-it"

    def enable_fairpro(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Enable FairPro system prompt generation.

        Note: Unlike QwenImagePipeline, Sana's built-in text encoder (Gemma) cannot
        be used directly for text generation because it's loaded without the LM head.
        This method always loads a separate model for FairPro generation.

        By default, uses "google/gemma-2-2b-it" which matches Sana's text encoder
        architecture but is loaded with the LM head for generation.

        Args:
            model_name: HuggingFace model name for the LLM. Defaults to
                "google/gemma-2-2b-it".
            device: Device to load the model on. If None, uses next available GPU
                or same device as pipeline.
            model: Pre-loaded model instance. If provided, model_name is ignored.
            tokenizer: Pre-loaded tokenizer instance. Required if model is provided.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Use default model name if not specified
        effective_model_name = model_name or self.DEFAULT_FAIRPRO_MODEL
        self._fairpro_model_name = effective_model_name
        self._use_builtin_encoder = False

        # Check if user provided a pre-loaded model
        if model is not None and tokenizer is not None:
            self._fairpro_model = model
            self._fairpro_tokenizer = tokenizer
            self._fairpro_enabled = True
            self._fairpro_model_name = "User-provided model"

            if hasattr(model, "device"):
                self._fairpro_device = str(model.device)
            elif device:
                self._fairpro_device = device
            else:
                self._fairpro_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            logger.info("FairPro enabled using user-provided model")
            logger.info(f"  Device: {self._fairpro_device}")
            return

        # Determine device
        if device is not None:
            self._fairpro_device = device
        elif hasattr(self, "_execution_device"):
            # Use the same device as the pipeline execution device
            exec_device = str(self._execution_device)
            if exec_device.startswith("cuda:"):
                # Check if next GPU is available, otherwise use same device
                gpu_id = int(exec_device.split(":")[1])
                next_gpu = gpu_id + 1
                if next_gpu < torch.cuda.device_count():
                    self._fairpro_device = f"cuda:{next_gpu}"
                else:
                    # Fall back to same device if no additional GPU available
                    self._fairpro_device = exec_device
            else:
                self._fairpro_device = exec_device
        else:
            self._fairpro_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the model with LM head for text generation
        logger.info(
            f"Loading FairPro LLM: {effective_model_name} on {self._fairpro_device}..."
        )
        self._fairpro_tokenizer = AutoTokenizer.from_pretrained(effective_model_name)
        self._fairpro_model = AutoModelForCausalLM.from_pretrained(
            effective_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self._fairpro_device},
        )
        self._fairpro_enabled = True
        logger.info(f"FairPro LLM loaded successfully on {self._fairpro_device}!")

    def disable_fairpro(self) -> None:
        """Disable FairPro and free the LLM memory."""
        self._fairpro_enabled = False

        if self._fairpro_model is not None:
            del self._fairpro_model
            self._fairpro_model = None
        if self._fairpro_tokenizer is not None:
            del self._fairpro_tokenizer
            self._fairpro_tokenizer = None

        torch.cuda.empty_cache()  # noqa: F821
        logger.info("FairPro disabled")

    def get_meta_prompt(self) -> str:
        """Return Sana-specific meta prompt template.

        Sana requires the system prompt to end with "User Prompt: " for proper
        prompt enhancement workflow.
        """
        return SANA_META_PROMPT_TEMPLATE.format(
            SANA_DEFAULT_SYSTEM_PROMPT="\n".join(SANA_DEFAULT_SYSTEM_PROMPT)
        )

    def get_default_system_prompt(self) -> str:
        """Get the default system prompt for Sana."""
        return "\n".join(SANA_DEFAULT_SYSTEM_PROMPT)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int | None = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: list[str] | None = None,
        # FairPro-specific arguments
        use_fairpro: bool = True,
        fairpro_system_prompts: str | list[str] | None = None,
    ) -> SanaPipelineOutput | tuple:
        """Generate images with optional FairPro fairness-aware system prompts.

        This method is API-compatible with the original SanaPipeline.__call__,
        with additional FairPro-specific arguments.

        Args:
            prompt: The prompt(s) for image generation.
            negative_prompt: Negative prompt for guidance.
            num_inference_steps: Number of denoising steps.
            timesteps: Custom timesteps for the scheduler.
            sigmas: Custom sigmas for the scheduler.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images to generate per prompt.
            height: Height of the generated image.
            width: Width of the generated image.
            eta: Eta parameter for DDIM scheduler.
            generator: Random number generator for reproducibility.
            latents: Pre-generated latents for image generation.
            prompt_embeds: Pre-computed prompt embeddings.
            prompt_attention_mask: Attention mask for prompt embeddings.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            negative_prompt_attention_mask: Attention mask for negative embeddings.
            output_type: Output format ("pil", "latent", "pt", "np").
            return_dict: Whether to return a pipeline output object.
            clean_caption: Whether to clean the caption.
            use_resolution_binning: Whether to use resolution binning.
            attention_kwargs: Additional attention arguments.
            callback_on_step_end: Callback function called at each step end.
            callback_on_step_end_tensor_inputs: Tensor inputs for callback.
            max_sequence_length: Maximum sequence length for text encoding.
            complex_human_instruction: Default human instruction for Sana.
            use_fairpro: Whether to use FairPro system prompt generation.
            fairpro_system_prompts: Pre-generated FairPro system prompt(s).
                Can be a single string (applied to all prompts) or a list
                (one per prompt).

        Returns:
            SanaPipelineOutput containing the generated images.
        """
        # Normalize prompt to list
        if prompt is not None and isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Handle FairPro system prompts
        fairpro_prompt = complex_human_instruction

        if use_fairpro and self._fairpro_enabled and prompt_embeds is None:
            # Generate or use provided system prompts
            system_prompts = self._normalize_system_prompts(
                fairpro_system_prompts, len(prompts)
            )
            if not system_prompts:
                # Generate FairPro system prompts for each prompt
                logger.info(
                    f"Generating FairPro system prompts for {len(prompts)} prompt(s)..."
                )
                system_prompts = self.generate_fairpro_system_prompts_batch(prompts)
                self._log_system_prompt_generation(prompts, system_prompts)
            else:
                logger.info(
                    "Using provided FairPro system prompts for image generation."
                )
            fairpro_prompt = system_prompts

        # Call parent pipeline
        return super().__call__(
            prompt=prompts if prompts else prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            output_type=output_type,
            return_dict=return_dict,
            clean_caption=clean_caption,
            use_resolution_binning=use_resolution_binning,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=fairpro_prompt,
        )

    def generate_comparison_images(
        self,
        prompt: str,
        seed: int | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        negative_prompt: str = "",
        guidance_scale: float = 4.5,
        **kwargs,
    ) -> tuple[Any, Any, str]:
        """Generate comparison images with default and FairPro system prompts.

        This is a convenience method for generating paired images that can be
        used for fairness comparison studies.

        Args:
            prompt: The user prompt for image generation.
            seed: Random seed for reproducibility. If None, uses random seed.
            height: Height of the generated images.
            width: Width of the generated images.
            num_inference_steps: Number of denoising steps.
            negative_prompt: Negative prompt for guidance.
            guidance_scale: Guidance scale.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            Tuple of (default_image, fairpro_image, fairpro_system_prompt).
        """
        if not self._fairpro_enabled:
            raise RuntimeError(
                "FairPro must be enabled to generate comparison images. "
                "Call enable_fairpro() first."
            )

        # Set up generator for reproducibility
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            generator = None

        gen_params = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **kwargs,
        }

        # Generate with default (no FairPro)
        default_image = self(use_fairpro=False, **gen_params).images[0]

        # Reset generator for same seed
        if seed is not None:
            gen_params["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        # Generate FairPro system prompt and image
        fairpro_system_prompt = self.generate_fairpro_system_prompt(prompt)
        fairpro_image = self(
            use_fairpro=True,
            fairpro_system_prompts=fairpro_system_prompt,
            **gen_params,
        ).images[0]

        return default_image, fairpro_image, fairpro_system_prompt

    def batch_generate_with_fairpro(
        self,
        prompts: list[str],
        seeds: list[int] | None = None,
        **kwargs,
    ) -> list[tuple[Any, str]]:
        """Generate images for multiple prompts with per-prompt FairPro system prompts.

        This method generates images one at a time, each with its own FairPro
        system prompt, which is useful when you need both the image and its
        corresponding system prompt.

        For batch generation without needing individual system prompts, use
        the regular __call__ method with a list of prompts.

        Args:
            prompts: List of user prompts for image generation.
            seeds: Optional list of seeds, one per prompt.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            List of tuples (image, fairpro_system_prompt) for each prompt.
        """
        if not self._fairpro_enabled:
            raise RuntimeError(
                "FairPro must be enabled for batch generation. Call enable_fairpro() first."
            )

        results = []
        for i, prompt in enumerate(prompts):
            seed = seeds[i] if seeds is not None else None

            # Set up generator
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)
            else:
                generator = None

            # Generate fairness-aware system prompt
            fairpro_system_prompt = self.generate_fairpro_system_prompt(prompt)

            # Generate image
            image = self(
                prompt=prompt,
                use_fairpro=True,
                fairpro_system_prompts=fairpro_system_prompt,
                generator=generator,
                **kwargs,
            ).images[0]

            results.append((image, fairpro_system_prompt))

            # Clear memory between batches
            torch.cuda.empty_cache()

        return results
