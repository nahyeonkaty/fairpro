"""FairPro: Fairness-aware system prompts for text-to-image generation.

FairPro provides pipelines and utilities for generating fairness-aware
system prompts that promote diversity and inclusivity in text-to-image models.

Example:
    >>> from fairpro import FairProQwenImagePipeline
    >>> pipe = FairProQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")
    >>> pipe.enable_fairpro()
    >>> image = pipe("a photo of a doctor").images[0]
"""

from fairpro.fairpro_mixin import FairProMixin
from fairpro.fairpro_qwenimage import FairProQwenImagePipeline
from fairpro.fairpro_sana import FairProSanaPipeline

__version__ = "0.0.1a1"

__all__ = [
    "FairProMixin",
    "FairProQwenImagePipeline",
    "FairProSanaPipeline",
]
