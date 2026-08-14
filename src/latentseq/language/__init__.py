"""Advanced language construction interfaces expose the semantic lowering pipeline.

Normal users construct `latentseq.Language` directly.  This namespace exists for researchers who
need to inspect, analyze, or modify the CFG, LS-PCFG, or compiled LanguageCore boundaries.
"""

from .analysis import LanguageAnalysis, analyze_language
from .cfg import ParsedCFG, parse_cfg, sample_cfg
from .compilation import LanguageCore, compile_language_core
from .ir import LSPCFG
from .sampling import sample_ls_pcfg

__all__ = [
    "LanguageAnalysis",
    "LanguageCore",
    "LSPCFG",
    "ParsedCFG",
    "analyze_language",
    "compile_language_core",
    "parse_cfg",
    "sample_cfg",
    "sample_ls_pcfg",
]
