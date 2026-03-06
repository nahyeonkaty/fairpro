"""FairPro Mixin providing fairness-aware system prompt generation functionality."""

from __future__ import annotations

import hashlib
import logging
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

# Bias check prompt template - asks the model to classify if a prompt could lead to social bias
BIAS_CHECK_PROMPT = """You are a helpful assistant that determines if an image generation prompt could lead to social bias.

A prompt could lead to social bias if it:
- Involves humans (e.g., people, professions, social roles)
- Could result in stereotypical representations of gender, race, age, or ethnicity
- Describes social situations where demographic representation matters

Analyze the following prompt and answer with ONLY "yes" or "no":
- Answer "yes" if the prompt could lead to social bias (e.g., involves humans)
- Answer "no" if the prompt is unlikely to lead to social bias (e.g., landscapes, objects, animals without human context)

Prompt: "{user_prompt}"

Answer (yes/no):"""

# Candidate scoring prompt template - scores fairness and faithfulness
CANDIDATE_SCORING_PROMPT = """You are evaluating a candidate system prompt for text-to-image generation.

User prompt:
"{user_prompt}"

Candidate system prompt:
"{system_prompt}"

Score the candidate on:
1) fairness (1-5): higher means less likely to induce social stereotypes.
2) faithfulness (1-5): higher means better preservation of user intent.

Return ONLY in this format:
fairness=<1-5>;faithfulness=<1-5>
"""


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
    _fairpro_prompt_cache: dict[tuple[Any, ...], str]
    _fairpro_bias_cache: dict[str, bool]

    @abstractmethod
    def enable_fairpro(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        quantization: str | None = None,
    ) -> None:
        """Enable FairPro system prompt generation.

        Args:
            model_name: HuggingFace model name for the LLM.
            device: Device to load the model on.
            model: Pre-loaded model instance.
            tokenizer: Pre-loaded tokenizer instance.
            quantization: Optional quantization mode when loading external model
                ("4bit", "8bit", or None).
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

    def _ensure_fairpro_caches(self) -> None:
        """Initialize in-memory caches lazily."""
        if not hasattr(self, "_fairpro_prompt_cache"):
            self._fairpro_prompt_cache = {}
        if not hasattr(self, "_fairpro_bias_cache"):
            self._fairpro_bias_cache = {}

    def clear_fairpro_cache(self) -> None:
        """Clear in-memory caches for bias checks and system prompts."""
        self._ensure_fairpro_caches()
        self._fairpro_prompt_cache.clear()
        self._fairpro_bias_cache.clear()

    def _build_prompt_cache_key(
        self,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        skip_bias_check: bool,
        num_candidates: int,
        select_best: bool,
        fairness_weight: float,
        faithfulness_weight: float,
    ) -> tuple[Any, ...]:
        """Build a stable cache key for a generated system prompt."""
        meta_prompt_hash = hashlib.md5(
            self.get_meta_prompt().encode("utf-8")
        ).hexdigest()
        return (
            user_prompt,
            max_new_tokens,
            round(temperature, 4),
            skip_bias_check,
            num_candidates,
            select_best,
            round(fairness_weight, 4),
            round(faithfulness_weight, 4),
            self._fairpro_model_name,
            meta_prompt_hash,
            self.__class__.__name__,
        )

    def _get_quantization_config(self, quantization: str | None) -> dict[str, Any]:
        """Build transformers quantization config kwargs.

        Args:
            quantization: Optional quantization mode ("4bit", "8bit", or None).

        Returns:
            Keyword arguments for `from_pretrained`.
        """
        if quantization is None:
            return {}

        quantization = quantization.lower()
        if quantization not in {"4bit", "8bit"}:
            raise ValueError(
                f"Unsupported quantization mode: {quantization}. "
                "Use one of: None, '4bit', '8bit'."
            )

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Quantization requires bitsandbytes. Install with `uv pip install bitsandbytes`."
            ) from exc

        if quantization == "4bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }

        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}

    def _chunk_list(self, items: list[Any], batch_size: int) -> list[list[Any]]:
        """Split list into contiguous chunks."""
        if batch_size <= 0:
            batch_size = 1
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

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

    def _generate_from_formatted_prompts(
        self,
        formatted_prompts: list[str],
        model: Any,
        tokenizer: Any,
        device: torch.device | str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate system prompts from already-formatted prompts in a true batch."""
        if not formatted_prompts:
            return []

        inputs = self._tokenize_for_generation(
            formatted_prompts=formatted_prompts,
            tokenizer=tokenizer,
            device=device,
            max_length=2048,
            use_chat_template=True,
        )

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).tolist()
        else:
            seq_len = int(inputs["input_ids"].shape[1])
            input_lengths = [seq_len] * len(formatted_prompts)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        if num_return_sequences > 1:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        if num_return_sequences > 1:
            expanded_lengths: list[int] = []
            for length in input_lengths:
                expanded_lengths.extend([int(length)] * num_return_sequences)
            input_lengths = expanded_lengths

        return self._decode_generated_outputs(outputs, input_lengths, tokenizer)

    def _parse_candidate_scores(self, response: str) -> tuple[float, float]:
        """Parse fairness/faithfulness scores from evaluator output."""
        fairness_match = re.search(r"fairness\s*[:=]\s*([1-5])", response, flags=re.I)
        faithfulness_match = re.search(
            r"faithfulness\s*[:=]\s*([1-5])",
            response,
            flags=re.I,
        )

        if fairness_match and faithfulness_match:
            return float(fairness_match.group(1)), float(faithfulness_match.group(1))

        fallback_scores = re.findall(r"\b([1-5])\b", response)
        if len(fallback_scores) >= 2:
            return float(fallback_scores[0]), float(fallback_scores[1])

        # Conservative fallback when parser fails.
        return 3.0, 3.0

    def _score_system_prompt_candidate(
        self,
        user_prompt: str,
        system_prompt: str,
        model: Any,
        tokenizer: Any,
        device: torch.device | str,
    ) -> tuple[float, float]:
        """Score one candidate system prompt on fairness and faithfulness."""
        scoring_prompt = CANDIDATE_SCORING_PROMPT.format(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        decoded = self._generate_from_formatted_prompts(
            formatted_prompts=[scoring_prompt],
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=24,
            temperature=0.0,
            do_sample=False,
            num_return_sequences=1,
        )[0]
        return self._parse_candidate_scores(decoded)

    def _select_best_system_prompt_candidate(
        self,
        user_prompt: str,
        candidates: list[str],
        model: Any,
        tokenizer: Any,
        device: torch.device | str,
        fairness_weight: float,
        faithfulness_weight: float,
    ) -> str:
        """Select best candidate using weighted fairness + faithfulness."""
        if not candidates:
            return self.get_default_system_prompt()

        deduped = list(dict.fromkeys(candidates))
        best_candidate = deduped[0]
        best_score = float("-inf")

        for candidate in deduped:
            fairness, faithfulness = self._score_system_prompt_candidate(
                user_prompt=user_prompt,
                system_prompt=candidate,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            combined = fairness_weight * fairness + faithfulness_weight * faithfulness
            logger.debug(
                "Candidate score (fairness=%.2f faithfulness=%.2f combined=%.2f): %s",
                fairness,
                faithfulness,
                combined,
                candidate[:120],
            )
            if combined > best_score:
                best_score = combined
                best_candidate = candidate

        return best_candidate

    def check_prompt_for_bias(
        self,
        user_prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_cache: bool = True,
    ) -> bool:
        """Check if a user prompt could lead to social bias.

        Uses the text model to classify if the prompt involves humans or
        could result in biased image generation.

        Args:
            user_prompt: The user's input prompt for image generation.
            max_new_tokens: Maximum tokens to generate (should be small for yes/no).
            temperature: Sampling temperature (used only when do_sample=True).
            do_sample: Whether to sample tokens. Defaults to False for deterministic
                yes/no classification.
            use_cache: Whether to reuse in-memory bias-check results.

        Returns:
            True if the prompt could lead to social bias (apply FairPro),
            False if no bias concern (use default system prompt).

        Raises:
            RuntimeError: If FairPro is not enabled.
        """
        if not self._fairpro_enabled:
            raise RuntimeError("FairPro is not enabled. Call enable_fairpro() first.")

        self._ensure_fairpro_caches()
        if use_cache and (not do_sample) and user_prompt in self._fairpro_bias_cache:
            return self._fairpro_bias_cache[user_prompt]

        model, tokenizer = self._get_fairpro_model_and_tokenizer()
        device = self._get_fairpro_device()

        # Format the bias check prompt
        formatted_prompt = BIAS_CHECK_PROMPT.format(user_prompt=user_prompt)

        # Tokenize
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

        # Generate response
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode and parse response
        response = (
            tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            )
            .strip()
            .lower()
        )

        logger.debug(f"Bias check response for '{user_prompt[:50]}...': {response}")

        # Parse yes/no response
        # Default to True (apply FairPro) if response is ambiguous
        result = not response.startswith("no")
        if use_cache and (not do_sample):
            self._fairpro_bias_cache[user_prompt] = result
        return result  # "yes" or ambiguous -> apply FairPro

    def generate_fairpro_system_prompt(
        self,
        user_prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        skip_bias_check: bool = False,
        batch_size: int = 8,
        use_cache: bool = True,
        num_candidates: int = 1,
        select_best: bool = False,
        fairness_weight: float = 0.6,
        faithfulness_weight: float = 0.4,
    ) -> str:
        """Generate a fairness-aware system prompt for a single user prompt.

        Args:
            user_prompt: The user's input prompt for image generation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature for generation.
            skip_bias_check: If True, always generate FairPro prompt.
            batch_size: Batch size used by internal generation path.
            use_cache: Whether to use in-memory cache.
            num_candidates: Number of candidate prompts to sample per user prompt.
            select_best: Whether to select best prompt by scoring candidates.
            fairness_weight: Weight for fairness score in candidate selection.
            faithfulness_weight: Weight for faithfulness score in candidate selection.

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
            skip_bias_check=skip_bias_check,
            batch_size=batch_size,
            use_cache=use_cache,
            num_candidates=num_candidates,
            select_best=select_best,
            fairness_weight=fairness_weight,
            faithfulness_weight=faithfulness_weight,
        )
        return results[0]

    def generate_fairpro_system_prompts_batch(
        self,
        user_prompts: list[str],
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        skip_bias_check: bool = False,
        batch_size: int = 8,
        use_cache: bool = True,
        num_candidates: int = 1,
        select_best: bool = False,
        fairness_weight: float = 0.6,
        faithfulness_weight: float = 0.4,
    ) -> list[str]:
        """Generate fairness-aware system prompts for multiple user prompts in a batch.

        This method first checks if each prompt could lead to social bias.
        If yes, it generates a FairPro system prompt. If no, it uses the
        default system prompt.

        Args:
            user_prompts: List of user prompts for image generation.
            max_new_tokens: Maximum number of tokens to generate per prompt.
            temperature: Sampling temperature for generation.
            skip_bias_check: If True, skip bias check and always generate
                FairPro system prompts (legacy behavior).
            batch_size: Number of prompts to generate in a single model call.
            use_cache: Whether to use in-memory cache for bias and prompt results.
            num_candidates: Number of sampled candidates per prompt.
            select_best: If True and num_candidates > 1, score and select
                best candidate via fairness + faithfulness.
            fairness_weight: Weight for fairness score during selection.
            faithfulness_weight: Weight for faithfulness score during selection.

        Returns:
            List of system prompts (FairPro or default based on bias check).

        Raises:
            RuntimeError: If FairPro is not enabled.
        """
        if not self._fairpro_enabled:
            raise RuntimeError("FairPro is not enabled. Call enable_fairpro() first.")

        if not user_prompts:
            return []

        self._ensure_fairpro_caches()

        model, tokenizer = self._get_fairpro_model_and_tokenizer()
        device = self._get_fairpro_device()

        # Get default system prompt for non-biased prompts
        default_system_prompt = self.get_default_system_prompt()

        # Format prompts with meta prompt template
        meta_prompt = self.get_meta_prompt()

        if batch_size <= 0:
            batch_size = 1
        num_candidates = max(1, int(num_candidates))

        weight_sum = fairness_weight + faithfulness_weight
        if weight_sum <= 0:
            fairness_weight = 0.5
            faithfulness_weight = 0.5
        else:
            fairness_weight = fairness_weight / weight_sum
            faithfulness_weight = faithfulness_weight / weight_sum

        system_prompts = [default_system_prompt] * len(user_prompts)
        generate_indices: list[int] = []
        generate_prompts: list[str] = []
        cache_keys: list[tuple[Any, ...]] = []

        for idx, user_prompt in enumerate(user_prompts):
            cache_key = self._build_prompt_cache_key(
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                skip_bias_check=skip_bias_check,
                num_candidates=num_candidates,
                select_best=select_best,
                fairness_weight=fairness_weight,
                faithfulness_weight=faithfulness_weight,
            )

            if use_cache and cache_key in self._fairpro_prompt_cache:
                system_prompts[idx] = self._fairpro_prompt_cache[cache_key]
                continue

            if not skip_bias_check:
                could_lead_to_bias = self.check_prompt_for_bias(
                    user_prompt,
                    use_cache=use_cache,
                )
                if not could_lead_to_bias:
                    logger.info(
                        "Prompt '%s...' -> No bias concern, using default system prompt",
                        user_prompt[:50],
                    )
                    system_prompts[idx] = default_system_prompt
                    if use_cache:
                        self._fairpro_prompt_cache[cache_key] = default_system_prompt
                    continue

            generate_indices.append(idx)
            generate_prompts.append(user_prompt)
            cache_keys.append(cache_key)

        if not generate_prompts:
            return system_prompts

        use_candidate_selector = select_best and num_candidates > 1

        if use_candidate_selector:
            for idx, user_prompt, cache_key in zip(
                generate_indices,
                generate_prompts,
                cache_keys,
            ):
                logger.info(
                    "Prompt '%s...' -> Generating %d candidates and selecting best",
                    user_prompt[:50],
                    num_candidates,
                )
                formatted = meta_prompt.format(user_prompt=user_prompt)
                candidates = self._generate_from_formatted_prompts(
                    formatted_prompts=[formatted],
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=num_candidates,
                )
                selected_prompt = self._select_best_system_prompt_candidate(
                    user_prompt=user_prompt,
                    candidates=candidates,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    fairness_weight=fairness_weight,
                    faithfulness_weight=faithfulness_weight,
                )
                system_prompts[idx] = selected_prompt
                if use_cache:
                    self._fairpro_prompt_cache[cache_key] = selected_prompt
            return system_prompts

        formatted_prompts = [
            meta_prompt.format(user_prompt=user_prompt)
            for user_prompt in generate_prompts
        ]
        generated_system_prompts: list[str] = []
        for chunk in self._chunk_list(formatted_prompts, batch_size):
            generated_system_prompts.extend(
                self._generate_from_formatted_prompts(
                    formatted_prompts=chunk,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1,
                )
            )

        for idx, cache_key, generated_prompt in zip(
            generate_indices,
            cache_keys,
            generated_system_prompts,
        ):
            system_prompts[idx] = generated_prompt
            if use_cache:
                self._fairpro_prompt_cache[cache_key] = generated_prompt

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
