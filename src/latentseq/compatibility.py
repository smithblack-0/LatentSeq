"""Compatibility checks make independently pooled LatentSeq components safe to recombine.

Agent compatibility is structural because different Agents are intentionally reusable with a common
Language. Vocabulary compatibility additionally carries the exact semantic Language fingerprint, so
matching tensor widths cannot accidentally authorize a vocabulary sampled for a different grammar.
"""

from .agent import Agent
from .language.api import Language
from .vocabulary import Vocabulary


# main


class CompatibilityError(ValueError):
    """Report one or more incompatibilities between independently reusable LatentSeq components."""


def validate_compatibility(*components: Agent | Language | Vocabulary) -> None:
    """Validate any supplied Agent, Language, and/or Vocabulary combination.

    Args:
        components: Two or three distinct component types in any order.

    Returns:
        None when all meaningful pairwise contracts hold. Raises `CompatibilityError` with all
        discovered mismatches otherwise.
    """
    if len(components) < 2 or len(components) > 3:
        raise TypeError("validate_compatibility expects two or three components")
    agent: Agent | None = None
    language: Language | None = None
    vocabulary: Vocabulary | None = None
    for component in components:
        if isinstance(component, Agent):
            if agent is not None:
                raise TypeError("only one Agent may be supplied")
            agent = component
        elif isinstance(component, Language):
            if language is not None:
                raise TypeError("only one Language may be supplied")
            language = component
        elif isinstance(component, Vocabulary):
            if vocabulary is not None:
                raise TypeError("only one Vocabulary may be supplied")
            vocabulary = component
        else:
            raise TypeError(f"unsupported compatibility component {type(component).__name__}")

    problems: list[str] = []
    if agent is not None and language is not None:
        if len(agent.cores) != language.num_cores:
            problems.append(
                "Agent/Language core count mismatch: "
                f"Agent has {len(agent.cores)}, Language expects {language.num_cores}."
            )
        else:
            for index, width in enumerate(agent.state_signature):
                if width > language.max_num_states:
                    problems.append(
                        "Agent/Language state width mismatch: "
                        f"Agent core {index} emits {width} states, Language supports "
                        f"{language.max_num_states}."
                    )

    if language is not None and vocabulary is not None:
        if vocabulary.language_fingerprint != language.fingerprint:
            problems.append(
                "Language/Vocabulary fingerprint mismatch: Vocabulary was sampled for a different "
                "Language identity."
            )
        if vocabulary.num_cores != language.num_cores:
            problems.append(
                "Language/Vocabulary core count mismatch: "
                f"Language has {language.num_cores}, Vocabulary expects {vocabulary.num_cores}."
            )
        if vocabulary.max_num_states != language.max_num_states:
            problems.append(
                "Language/Vocabulary state width mismatch: "
                f"Language has {language.max_num_states}, Vocabulary expects "
                f"{vocabulary.max_num_states}."
            )
        if vocabulary.grammar_terminal_count != language.grammar_terminal_count:
            problems.append(
                "Language/Vocabulary terminal width mismatch: "
                f"Language emits {language.grammar_terminal_count} terminals, Vocabulary expects "
                f"{vocabulary.grammar_terminal_count}."
            )

    if agent is not None and vocabulary is not None:
        if len(agent.cores) != vocabulary.num_cores:
            problems.append(
                "Agent/Vocabulary core count mismatch: "
                f"Agent has {len(agent.cores)}, Vocabulary expects {vocabulary.num_cores}."
            )
        else:
            for index, width in enumerate(agent.state_signature):
                if width > vocabulary.max_num_states:
                    problems.append(
                        "Agent/Vocabulary state width mismatch: "
                        f"Agent core {index} emits {width} states, Vocabulary supports "
                        f"{vocabulary.max_num_states}."
                    )

    if problems:
        raise CompatibilityError("Compatibility check failed:\n  - " + "\n  - ".join(problems))
