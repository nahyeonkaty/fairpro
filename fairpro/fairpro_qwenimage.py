"""FairPro QwenImage Pipeline with per-prompt fairness-aware system prompt generation."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from diffusers import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    QwenImagePipelineOutput,
)

from fairpro.fairpro_mixin import FairProMixin

logger = logging.getLogger(__name__)

# Prompt template for Qwen-Image models
# Note: {{}} is escaped so first .format(system_prompt=...) leaves {} for user prompt
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n{{}}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Default start index for the default system prompt (matches original pipeline)
DEFAULT_PROMPT_TEMPLATE_START_IDX = 34

# QwenImage default system prompt (matches diffusers QwenImagePipeline)
QWENIMAGE_DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
)

# QwenImage-specific meta prompt template
QWENIMAGE_META_PROMPT_TEMPLATE = """You are an intelligent, unbiased assistant.
Your goal is to design a fair instruction that guides detailed, accurate, and globally inclusive image descriptions.

Consider the following user prompt:
"{user_prompt}"

Identify what would be the problem if the original system prompt were used for the above user prompt. Focus on:
- Potential social stereotypes or biases that might emerge
- Lack of diversity or inclusivity
- Cultural or demographic assumptions

Think of a concise analysis of the potential issues.
Based on this analysis, create a system prompt that addresses these issues and promotes fairness, diversity, and inclusivity.

CAUTION: If the user prompt does not involve humans (e.g., inanimate objects, landscapes, animals), you MUST output the 'Original system prompt' exactly as written below, word-for-word.
Do not add any extra clauses about fairness, diversity, or inclusivity to this string.
Original system prompt: "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"

Output format:
[Write only the final revised system prompt here—no explanations, no reasoning text.]
"""


class FairProQwenImagePipeline(FairProMixin, QwenImagePipeline):
    """Qwen Image Pipeline with FairPro fairness-aware system prompt generation.

    This pipeline extends QwenImagePipeline to automatically generate and apply
    fairness-aware system prompts for each user prompt before image generation.

    Key features:
    - Uses QwenImage's built-in Qwen2.5 text encoder for system prompt generation
    - Supports per-prompt system prompts (different system prompt for each prompt in batch)
    - API compatible with original QwenImagePipeline

    Example:
        ```python
        from pipelines import FairProQwenImagePipeline

        # Load the pipeline
        pipe = FairProQwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )

        # Enable FairPro functionality (uses built-in text encoder)
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

    # Default model name that uses the built-in text encoder
    DEFAULT_FAIRPRO_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

    def enable_fairpro(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Enable FairPro system prompt generation.

        By default, uses QwenImage's built-in Qwen2.5-VL text encoder for generating
        fairness-aware system prompts, avoiding the need to load a separate model.

        If a different model_name is specified, it will load that model separately.

        Args:
            model_name: HuggingFace model name for the LLM. Defaults to
                "Qwen/Qwen2.5-VL-7B-Instruct" which uses the built-in encoder.
                Specify a different model to load it separately.
            device: Device to load the model on (only used when loading separate model).
            model: Pre-loaded model instance. If provided, uses this instead.
            tokenizer: Pre-loaded tokenizer instance. Required if model is provided.
        """
        # Use default model name if not specified
        effective_model_name = model_name or self.DEFAULT_FAIRPRO_MODEL

        # Check if user provided a pre-loaded model
        if model is not None and tokenizer is not None:
            self._fairpro_model = model
            self._fairpro_tokenizer = tokenizer
            self._fairpro_enabled = True
            self._fairpro_model_name = "User-provided model"
            self._use_builtin_encoder = False

            if hasattr(model, "device"):
                self._fairpro_device = str(model.device)
            elif device:
                self._fairpro_device = device
            else:
                self._fairpro_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            logger.info("FairPro enabled using user-provided model")
            logger.info(f"  Device: {self._fairpro_device}")
            return

        # Check if using the default model (built-in text encoder)
        if effective_model_name == self.DEFAULT_FAIRPRO_MODEL:
            self._fairpro_model = self.text_encoder
            self._fairpro_tokenizer = self.tokenizer
            self._fairpro_enabled = True
            self._fairpro_model_name = "QwenImage built-in text encoder"
            self._use_builtin_encoder = True

            if hasattr(self.text_encoder, "device"):
                self._fairpro_device = str(self.text_encoder.device)
            else:
                self._fairpro_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            logger.info("FairPro enabled using QwenImage's built-in text encoder")
            logger.info(f"  Device: {self._fairpro_device}")
        else:
            # Load a separate model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._fairpro_model_name = effective_model_name
            self._use_builtin_encoder = False

            # Determine device
            if device is not None:
                self._fairpro_device = device
            elif hasattr(self, "device"):
                self._fairpro_device = str(self.device)
            else:
                self._fairpro_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"Loading FairPro LLM: {effective_model_name} on {self._fairpro_device}..."
            )
            self._fairpro_tokenizer = AutoTokenizer.from_pretrained(
                effective_model_name
            )
            self._fairpro_model = AutoModelForCausalLM.from_pretrained(
                effective_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self._fairpro_device},
            )
            self._fairpro_enabled = True
            logger.info(f"FairPro LLM loaded successfully on {self._fairpro_device}!")

    def get_default_system_prompt(self) -> str:
        """Get the default system prompt for QwenImage."""
        return QWENIMAGE_DEFAULT_SYSTEM_PROMPT

    def get_meta_prompt(self) -> str:
        """Return QwenImage-specific meta prompt template."""
        return QWENIMAGE_META_PROMPT_TEMPLATE

    def _get_fairpro_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Return the model and tokenizer for FairPro generation.

        QwenImage uses the built-in text encoder by default.
        """
        if self._use_builtin_encoder:
            return self.text_encoder, self.tokenizer
        return self._fairpro_model, self._fairpro_tokenizer

    def _get_fairpro_device(self) -> torch.device | str:
        """Return the device for FairPro generation."""
        if self._use_builtin_encoder:
            return self._execution_device
        return super()._get_fairpro_device()

    def _format_chat_message(self, content: str) -> dict:
        """Format chat message for text-only generation.

        For FairPro system prompt generation, we only use text (no images),
        so plain string content works with both tokenizer and processor.
        """
        return {"role": "user", "content": content}

    def _tokenize_for_generation(
        self,
        formatted_prompts: list[str],
        tokenizer: Any,
        device: torch.device | str,
        max_length: int = 2048,
        use_chat_template: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize prompts for batch generation.

        Uses the tokenizer's apply_chat_template with plain string content.
        Each prompt becomes a separate conversation with a single user message.
        """
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            # Each prompt is a separate conversation: [[msg1], [msg2], ...]
            batch_messages = [[self._format_chat_message(p)] for p in formatted_prompts]
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                add_generation_prompt=True,
            )
        else:
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        return {k: v.to(device) for k, v in inputs.items()}

    def disable_fairpro(self) -> None:
        """Disable FairPro functionality.

        If using the built-in text encoder, this only disables FairPro without
        unloading the encoder. If a separate model was loaded, it frees that memory.
        """
        self._fairpro_enabled = False

        # Only free memory if a separate model was loaded
        if hasattr(self, "_use_builtin_encoder") and not self._use_builtin_encoder:
            if self._fairpro_model is not None:
                del self._fairpro_model
            if self._fairpro_tokenizer is not None:
                del self._fairpro_tokenizer
            torch.cuda.empty_cache()

        self._fairpro_model = None
        self._fairpro_tokenizer = None
        logger.info("FairPro disabled")

    def _get_prompt_template_for_system_prompt(
        self, system_prompt: str
    ) -> tuple[str, int]:
        """Get prompt template and start index for a given system prompt.

        Args:
            system_prompt: The system prompt to use.

        Returns:
            Tuple of (prompt_template, start_idx).
        """
        template = PROMPT_TEMPLATE.format(system_prompt=system_prompt)
        start_idx = len(self.tokenizer.encode(template)) - 6
        return template, start_idx

    def _get_qwen_prompt_embeds_with_system_prompts(
        self,
        prompts: list[str],
        system_prompts: list[str],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get prompt embeddings with per-prompt system prompts.

        This method allows different system prompts for each prompt in a batch.

        Args:
            prompts: List of user prompts.
            system_prompts: List of system prompts, one per user prompt.
            device: Device to place tensors on.
            dtype: Data type for tensors.

        Returns:
            Tuple of (prompt_embeds, encoder_attention_mask).
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        assert len(prompts) == len(system_prompts), (
            f"Number of prompts ({len(prompts)}) must match system prompts ({len(system_prompts)})"
        )

        all_hidden_states = []
        all_attn_masks = []

        for prompt, system_prompt in zip(prompts, system_prompts):
            template, drop_idx = self._get_prompt_template_for_system_prompt(
                system_prompt
            )
            txt = template.format(prompt)

            txt_tokens = self.tokenizer(
                txt,
                max_length=self.tokenizer_max_length + drop_idx,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            encoder_hidden_states = self.text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )

            hidden_states = encoder_hidden_states.hidden_states[-1]
            split_hidden_states = self._extract_masked_hidden(
                hidden_states, txt_tokens.attention_mask
            )
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

            all_hidden_states.extend(split_hidden_states)
            for e in split_hidden_states:
                all_attn_masks.append(
                    torch.ones(e.size(0), dtype=torch.long, device=device)
                )

        max_seq_len = max(e.size(0) for e in all_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in all_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in all_attn_masks
            ]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        # FairPro-specific arguments
        use_fairpro: bool = True,
        fairpro_system_prompts: str | list[str] | None = None,
    ) -> QwenImagePipelineOutput | tuple:
        """Generate images with optional FairPro fairness-aware system prompts.

        This method is API-compatible with the original QwenImagePipeline.__call__,
        with additional FairPro-specific arguments.

        Args:
            prompt: The prompt(s) for image generation.
            negative_prompt: Negative prompt(s) for guidance.
            true_cfg_scale: True CFG scale for classifier-free guidance.
            height: Height of the generated image.
            width: Width of the generated image.
            num_inference_steps: Number of denoising steps.
            sigmas: Custom sigmas for the scheduler.
            guidance_scale: Guidance scale (for future guidance-distilled models).
            num_images_per_prompt: Number of images per prompt.
            generator: Random number generator for reproducibility.
            latents: Pre-generated latents.
            prompt_embeds: Pre-computed prompt embeddings.
            prompt_embeds_mask: Mask for prompt embeddings.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            negative_prompt_embeds_mask: Mask for negative prompt embeddings.
            output_type: Output format ("pil", "latent", "np").
            return_dict: Whether to return a pipeline output object.
            attention_kwargs: Additional attention arguments.
            callback_on_step_end: Callback function at each step.
            callback_on_step_end_tensor_inputs: Tensor inputs for callback.
            max_sequence_length: Maximum sequence length for prompts.
            use_fairpro: Whether to use FairPro system prompt generation.
            fairpro_system_prompts: Pre-generated FairPro system prompt(s).
                Can be a single string (applied to all prompts) or a list
                (one per prompt).

        Returns:
            QwenImagePipelineOutput containing the generated images.
        """
        # Normalize prompt to list
        if prompt is not None and isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Handle FairPro system prompts
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

            # Generate embeddings with per-prompt system prompts
            prompt_embeds, prompt_embeds_mask = (
                self._get_qwen_prompt_embeds_with_system_prompts(
                    prompts=prompts,
                    system_prompts=system_prompts,
                    device=self._execution_device,
                )
            )

            # Repeat for num_images_per_prompt
            if num_images_per_prompt > 1:
                batch_size = len(prompts)
                seq_len = prompt_embeds.shape[1]
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )
                prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
                prompt_embeds_mask = prompt_embeds_mask.view(
                    batch_size * num_images_per_prompt, seq_len
                )

        # Call parent pipeline with pre-computed embeddings or default behavior
        return super().__call__(
            prompt=prompt if prompt_embeds is None else None,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

    def generate_comparison_images(
        self,
        prompt: str,
        seed: int | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        negative_prompt: str = "low quality, worst quality, blurry, ugly",
        true_cfg_scale: float = 4.0,
        **kwargs,
    ) -> tuple[Any, Any, str]:
        """Generate comparison images with default and FairPro system prompts.

        Args:
            prompt: The user prompt for image generation.
            seed: Random seed for reproducibility.
            height: Height of the generated images.
            width: Width of the generated images.
            num_inference_steps: Number of denoising steps.
            negative_prompt: Negative prompt for guidance.
            true_cfg_scale: True CFG scale.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            Tuple of (default_image, fairpro_image, fairpro_system_prompt).
        """
        if not self._fairpro_enabled:
            raise RuntimeError("FairPro must be enabled. Call enable_fairpro() first.")

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
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
            **kwargs,
        }

        # Generate with default system prompt
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
            raise RuntimeError("FairPro must be enabled. Call enable_fairpro() first.")

        results = []
        for i, prompt in enumerate(prompts):
            seed = seeds[i] if seeds is not None else None

            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)
            else:
                generator = None

            fairpro_system_prompt = self.generate_fairpro_system_prompt(prompt)

            image = self(
                prompt=prompt,
                use_fairpro=True,
                fairpro_system_prompts=fairpro_system_prompt,
                generator=generator,
                **kwargs,
            ).images[0]

            results.append((image, fairpro_system_prompt))
            torch.cuda.empty_cache()

        return results
