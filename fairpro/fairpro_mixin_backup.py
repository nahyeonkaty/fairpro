"""FairPro Mixin providing fairness-aware system prompt generation functionality."""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


class FairProMixin:
    """Mixin class providing FairPro system prompt generation functionality.

    This mixin can be combined with any text-to-image pipeline to add
    fairness-aware system prompt generation capabilities.

    Subclasses must implement:
        - enable_fairpro(): Set up the FairPro model/tokenizer
        - disable_fairpro(): Clean up resources
        - get_default_system_prompt(): Return the default system prompt
        - get_meta_prompt(): Return the meta prompt template for system prompt generation

    Subclasses may override:
        - _get_fairpro_model_and_tokenizer(): Custom model/tokenizer setup
        - _get_fairpro_device(): Custom device handling
    """

    _fairpro_model: "AutoModelForCausalLM | PreTrainedModel | None" = None
    _fairpro_tokenizer: "AutoTokenizer | None" = None
    _fairpro_enabled: bool = False
    _fairpro_device: str = "cuda:0"
    _fairpro_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    _use_builtin_encoder: bool = False

    @abstractmethod
    def enable_fairpro(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Enable FairPro system prompt generation.

        Args:
            model_name: HuggingFace model name for the LLM.
            device: Device to load the model on.
            model: Pre-loaded model instance.
            tokenizer: Pre-loaded tokenizer instance.
        """
        raise NotImplementedError("Subclasses must implement enable_fairpro().")

    @abstractmethod
    def disable_fairpro(self) -> None:
        """Disable FairPro and free the LLM memory."""
        raise NotImplementedError("Subclasses must implement disable_fairpro().")

    @abstractmethod
    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for this pipeline."""
        raise NotImplementedError(
            "Subclasses must implement get_default_system_prompt()."
        )

    @abstractmethod
    def get_meta_prompt(self) -> str:
        """Return the meta prompt template for generating fairness-aware system prompts.

        The template must contain a {user_prompt} placeholder.

        Returns:
            A string template with {user_prompt} placeholder.
        """
        raise NotImplementedError("Subclasses must implement get_meta_prompt().")

    @property
    def fairpro_enabled(self) -> bool:
        """Check if FairPro is currently enabled."""
        return self._fairpro_enabled

    def _get_fairpro_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Return the model and tokenizer to use for FairPro generation.

        Override this method if your pipeline uses a different model/tokenizer
        setup (e.g., built-in text encoder).

        Returns:
            Tuple of (model, tokenizer).
        """
        return self._fairpro_model, self._fairpro_tokenizer

    def _get_fairpro_device(self) -> torch.device | str:
        """Return the device to use for FairPro generation.

        Override this method if your pipeline needs custom device handling.
        """
        model, _ = self._get_fairpro_model_and_tokenizer()
        if model is not None and hasattr(model, "device"):
            return model.device
        return self._fairpro_device

    def _parse_system_prompt(self, response: str) -> str:
        """Parse the system prompt from the model's response.

        Args:
            response: The raw model response.

        Returns:
            The extracted system prompt, or the default if extraction fails.
        """
        # Extract from tags if present
        if "<system_prompt>" in response and "</system_prompt>" in response:
            start_idx = response.find("<system_prompt>") + len("<system_prompt>")
            end_idx = response.find("</system_prompt>")
            extracted = response[start_idx:end_idx]
        else:
            extracted = response

        cleaned = extracted.strip()

        # Remove leading quote + backslash artifacts (e.g., '"\' or "'\\")
        if cleaned.startswith('"\\'):
            cleaned = cleaned[2:].lstrip()
        if cleaned.startswith("'\\"):
            cleaned = cleaned[2:].lstrip()

        # Remove wrapping quotes if present
        for quote in ['"""', "'''", '"', "'"]:
            if cleaned.startswith(quote) and cleaned.endswith(quote):
                cleaned = cleaned[len(quote) : -len(quote)].strip()

        # Remove literal escaped characters (appearing as text, not actual escapes)
        cleaned = cleaned.replace("\\n", " ").replace("\\t", " ")

        # Clean up whitespace: collapse multiple spaces and newlines
        cleaned = re.sub(
            r"[ \t]+", " ", cleaned
        )  # Multiple spaces/tabs -> single space
        cleaned = re.sub(
            r"\n\s*\n+", "\n", cleaned
        )  # Multiple newlines -> single newline

        # Trim each line (remove trailing spaces)
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
        cleaned = cleaned.strip()

        if not cleaned:
            return self.get_default_system_prompt()

        return cleaned

    def _format_chat_message(self, content: str) -> dict:
        """Format a single chat message for the model.

        Override this method for model-specific message formats.
        For example, Qwen2.5-VL expects content as a list of typed blocks:
            {"role": "user", "content": [{"type": "text", "text": "..."}]}

        Args:
            content: The text content of the message.

        Returns:
            A dictionary representing the chat message.
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

        Args:
            formatted_prompts: List of formatted prompt strings.
            tokenizer: The tokenizer to use.
            device: Device to place tensors on.
            max_length: Maximum sequence length.
            use_chat_template: Whether to apply chat template if available.

        Returns:
            Dictionary of tokenized inputs ready for generation.
        """
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            batch_messages = [[self._format_chat_message(p)] for p in formatted_prompts]
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=True,
                max_length=max_length,
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

    def _decode_generated_outputs(
        self,
        outputs: torch.Tensor,
        input_lengths: list[int],
        tokenizer: Any,
    ) -> list[str]:
        """Decode model outputs to system prompts.

        Args:
            outputs: Generated token sequences.
            input_lengths: Original input lengths for each sequence.
            tokenizer: The tokenizer to use for decoding.

        Returns:
            List of parsed system prompts.
        """
        system_prompts = []
        for output, input_len in zip(outputs, input_lengths):
            response = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            ).strip()
            logger.debug(
                f"Raw generated response (len={len(output[input_len:])}): {response[:200]}..."
            )
            parsed = self._parse_system_prompt(response)
            logger.debug(f"Parsed system prompt: {parsed[:100]}...")
            system_prompts.append(parsed)
        return system_prompts

    def generate_fairpro_system_prompt(
        self,
        user_prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
    ) -> str:
        """Generate a fairness-aware system prompt for a single user prompt.

        Args:
            user_prompt: The user's input prompt for image generation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature for generation.

        Returns:
            The generated fairness-aware system prompt.

        Raises:
            RuntimeError: If FairPro is not enabled.
        """
        if not self._fairpro_enabled:
            raise RuntimeError("FairPro is not enabled. Call enable_fairpro() first.")

        # Use batch method for single prompt (avoids code duplication)
        results = self.generate_fairpro_system_prompts_batch(
            [user_prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return results[0]

    def generate_fairpro_system_prompts_batch(
        self,
        user_prompts: list[str],
        max_new_tokens: int = 300,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate fairness-aware system prompts for multiple user prompts in a batch.

        This is the core batch generation method that handles tokenization,
        generation, and decoding efficiently.

        Args:
            user_prompts: List of user prompts for image generation.
            max_new_tokens: Maximum number of tokens to generate per prompt.
            temperature: Sampling temperature for generation.

        Returns:
            List of generated fairness-aware system prompts.

        Raises:
            RuntimeError: If FairPro is not enabled.
        """
        if not self._fairpro_enabled:
            raise RuntimeError("FairPro is not enabled. Call enable_fairpro() first.")

        if not user_prompts:
            return []

        model, tokenizer = self._get_fairpro_model_and_tokenizer()
        device = self._get_fairpro_device()

        # Format prompts with meta prompt template
        meta_prompt = self.get_meta_prompt()
        formatted_prompts = [meta_prompt.format(user_prompt=p) for p in user_prompts]

        # Process one prompt at a time to avoid batch tokenization issues
        system_prompts = []
        for formatted_prompt in formatted_prompts:
            # Tokenize single prompt
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [self._format_chat_message(formatted_prompt)]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)
            else:
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)

            input_length = inputs["input_ids"].shape[1]

            # Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode
            response = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            ).strip()
            logger.debug(f"Raw response: {response[:200]}...")
            system_prompts.append(self._parse_system_prompt(response))

        return system_prompts

    def _normalize_system_prompts(
        self,
        fairpro_system_prompts: str | list[str] | None,
        num_prompts: int,
    ) -> list[str]:
        """Normalize system prompts to a list matching the number of prompts.

        Args:
            fairpro_system_prompts: Single prompt, list of prompts, or None.
            num_prompts: Number of user prompts.

        Returns:
            List of system prompts with length matching num_prompts.
        """
        if fairpro_system_prompts is None:
            return []
        if isinstance(fairpro_system_prompts, str):
            return [fairpro_system_prompts] * num_prompts
        return list(fairpro_system_prompts)

    def _log_system_prompt_generation(
        self,
        prompts: list[str],
        system_prompts: list[str],
        verbose: bool = True,
    ) -> None:
        """Log system prompt generation details.

        Args:
            prompts: Original user prompts.
            system_prompts: Generated system prompts.
            verbose: Whether to log detailed info.
        """
        if not verbose:
            return

        for i, (p, sp) in enumerate(zip(prompts, system_prompts)):
            logger.info(f"[{i + 1}/{len(prompts)}] {p[:50]}...")
            logger.info(f"  -> {sp[:80]}...")
