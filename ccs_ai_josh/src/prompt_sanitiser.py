"""
Prompt sanitisation module.

All user-facing text inputs passed to the LLM are routed through
:func:`sanitise_input` before entering the conversation graph.

Sanitisation strategy
---------------------
1. **Type validation** – rejects non-string values with :class:`TypeError`.
2. **Length validation** – silently truncates inputs that exceed
   :data:`MAX_INPUT_LENGTH` characters to limit token-stuffing attacks.
3. **Injection pattern detection** – raises :class:`ValueError` when the input
   matches any of the patterns in :data:`INJECTION_PATTERNS`.  Patterns cover:

   * *Direct injection* – explicit instruction-override phrases such as
     "Ignore previous instructions …".
   * *SQL injection* – SQL execution keywords (``DROP TABLE``, ``SELECT … FROM
     … WHERE``, ``execute … sql``, etc.).
   * *Data-exfiltration attempts* – requests to reveal system prompts, API
     keys, secrets, or internal configuration.
   * *Jailbreak triggers* – well-known phrases used to bypass model safety
     constraints (e.g. "DAN mode", "developer mode").

Indirect injection via retrieved documents is addressed separately in the
generation step: see :func:`~ccs_ai_josh.src.multiturn_utils.generate`, which
wraps retrieved content in explicit ``BEGIN/END RETRIEVED DOCUMENTS`` delimiters
so that the LLM can distinguish external data from trusted instructions.

Reviewed by: (add reviewer name on approval)
"""

import re
import logging
from typing import Final

logger = logging.getLogger(__name__)

# Maximum number of characters accepted from a single user message.
MAX_INPUT_LENGTH: Final[int] = 2000

# Each entry is a (regex_pattern, human_readable_label) pair.
# Patterns are matched case-insensitively against the raw user string.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Direct instruction-override injection
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "direct injection"),
    (r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", "direct injection"),
    (r"forget\s+(all\s+)?(previous|prior|above)\s+instructions", "direct injection"),
    (r"override\s+(all\s+)?(previous|prior|above)\s+instructions", "direct injection"),
    # SQL execution attempts
    (r"execute\s+(the\s+)?(following\s+)?sql", "SQL execution"),
    (r"\b(drop|truncate)\s+table\b", "SQL injection"),
    (r"\bselect\b.{0,100}\bfrom\b.{0,100}\bwhere\b", "SQL injection"),
    # Data-exfiltration attempts
    (r"(reveal|show|print|tell me|display|expose|leak)\s+(your\s+)?(system\s+)?prompt", "data exfiltration"),
    (r"(reveal|show|tell me|expose|leak)\s+(me\s+)?(your\s+)?(api\s+key|secret|credential|password)", "data exfiltration"),
    (r"what\s+(is|are)\s+your\s+(system\s+)?(instruction|prompt|configuration|secret)", "data exfiltration"),
    # Jailbreak triggers
    (r"\bdan\s+mode\b", "jailbreak"),
    (r"\bdeveloper\s+mode\b", "jailbreak"),
]

# Pre-compile all patterns once at module load time for efficiency.
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), label)
    for pattern, label in INJECTION_PATTERNS
]


def sanitise_input(user_input: str) -> str:
    """Sanitise a raw user string before it is forwarded to the LLM.

    Parameters
    ----------
    user_input:
        The raw text entered by the user.

    Returns
    -------
    str
        The (possibly truncated) sanitised input string.

    Raises
    ------
    TypeError
        If *user_input* is not a :class:`str`.
    ValueError
        If *user_input* matches a known prompt-injection or SQL-injection
        pattern.
    """
    if not isinstance(user_input, str):
        raise TypeError(
            f"Expected string input, got {type(user_input).__name__}"
        )

    if len(user_input) > MAX_INPUT_LENGTH:
        logger.warning(
            "User input truncated from %d to %d characters",
            len(user_input),
            MAX_INPUT_LENGTH,
        )
        user_input = user_input[:MAX_INPUT_LENGTH]

    for pattern, label in _COMPILED_PATTERNS:
        if pattern.search(user_input):
            logger.warning("Potential %s detected in user input", label)
            raise ValueError(
                "Your input contains content that cannot be processed. "
                "Please rephrase your question."
            )

    return user_input
