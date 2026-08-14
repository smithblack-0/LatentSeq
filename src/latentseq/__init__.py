"""LatentSeq public API for reusable latent, language, and vocabulary process generation.

Normal usage intentionally exposes only the independently reusable components, optional Generator
orchestration, and compatibility validation. Semantic construction IRs and runtime helpers remain in
submodules so package autocomplete reflects the intended external abstraction layer.
"""

from .agent import Agent
from .compatibility import CompatibilityError, validate_compatibility
from .generator import GeneratedSample, Generator
from .language.api import Language
from .vocabulary import Vocabulary

__all__ = [
    "Agent",
    "CompatibilityError",
    "GeneratedSample",
    "Generator",
    "Language",
    "Vocabulary",
    "validate_compatibility",
]
