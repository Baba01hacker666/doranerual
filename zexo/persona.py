"""Zexo AI: Persona, Dialogue Formats & Conversational Templates.

Defines the system identity, behavior rules, and multi-turn prompt templates
for Zexo AI conversations.
"""

from typing import List, Optional


ZEXO_SYSTEM_PROMPT = (
    "You are Zexo, an intelligent, thoughtful, and creative conversational AI assistant. "
    "You communicate with clarity, warmth, and precision. You love helping users explore ideas, "
    "solve challenging problems, explain complex topics simply, and hold engaging, meaningful conversations."
)

ZEXO_STOP_SEQUENCES: List[str] = [
    "\nUser:",
    "User:",
    "\nHuman:",
    "Human:",
    "\nSystem:",
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
]


def format_zexo_dialogue(
    messages: List[dict],
    system_prompt: Optional[str] = None,
    pending_user_input: Optional[str] = None,
) -> str:
    """Format structured chat turns into a cohesive text prompt for Zexo.

    Args:
        messages: List of dicts with 'role' ('system', 'user', 'assistant') and 'content'.
        system_prompt: System persona instruction string.
        pending_user_input: Optional next user turn awaiting response.

    Returns:
        Formatted prompt string ending with 'Assistant:' ready for generation.
    """
    sys_instruction = system_prompt or ZEXO_SYSTEM_PROMPT
    lines = [f"System: {sys_instruction.strip()}"]

    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "").strip()
        if role == "user":
            lines.append(f"User: {content}")
        elif role in ("assistant", "zexo"):
            lines.append(f"Zexo: {content}")
        elif role == "system":
            lines.append(f"System: {content}")

    if pending_user_input:
        lines.append(f"User: {pending_user_input.strip()}")

    lines.append("Zexo:")
    return "\n".join(lines)
