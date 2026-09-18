#!/usr/bin/env python3
"""Assemble the legacy Python dialogue sources into a deterministic text file.

Run either ``python zexo/data/assemble_dataset.py`` or
``python -m zexo.data.assemble_dataset`` from any working directory.
For the newer quality-controlled corpus, use ``python -m zexo.data.dataset_tools``.
"""

from pathlib import Path

try:  # Package/module execution.
    from .dialogues_persona import DIALOGUES_PERSONA
    from .dialogues_python import DIALOGUES_PYTHON
    from .dialogues_deeplearning import DIALOGUES_DEEPLEARNING
    from .dialogues_mathphysics import DIALOGUES_MATHPHYSICS
    from .dialogues_practical import DIALOGUES_PRACTICAL
    from .dialogues_reasoning import DIALOGUES_REASONING
except ImportError:  # Direct script execution.
    from dialogues_persona import DIALOGUES_PERSONA
    from dialogues_python import DIALOGUES_PYTHON
    from dialogues_deeplearning import DIALOGUES_DEEPLEARNING
    from dialogues_mathphysics import DIALOGUES_MATHPHYSICS
    from dialogues_practical import DIALOGUES_PRACTICAL
    from dialogues_reasoning import DIALOGUES_REASONING


OUTPUT_FILE = Path(__file__).resolve().parent / "step1_conversational_expanded.txt"

ALL_CATEGORIES = [
    ("Persona & Identity", DIALOGUES_PERSONA),
    ("Python & Algorithms", DIALOGUES_PYTHON),
    ("Deep Learning & LLMs", DIALOGUES_DEEPLEARNING),
    ("Math & Physics", DIALOGUES_MATHPHYSICS),
    ("Practical Assistance & Systems", DIALOGUES_PRACTICAL),
    ("Logic, Algorithms & Reasoning", DIALOGUES_REASONING),
]

FORBIDDEN_KEYWORDS = [
    "once upon a time", "fairy tale", "goldilocks", "cinderella", "nursery rhyme",
    "humpty dumpty", "jack and jill", "sleeping beauty", "peter pan", "little red riding hood",
    "three little pigs", "snow white", "hansel and gretel", "rapunzel", "rumplestiltskin",
    "teletubbies", "peppa pig", "baby shark", "storybook",
]


def main() -> None:
    total_dialogues = 0
    total_turns = 0
    formatted_blocks = []

    print("=" * 60)
    print(" Validating and Assembling Zexo Conversational Dataset")
    print("=" * 60)

    for cat_name, dialogues in ALL_CATEGORIES:
        print(f"\nProcessing Category: {cat_name} ({len(dialogues)} dialogues)")
        for d_idx, dialogue in enumerate(dialogues, 1):
            total_dialogues += 1
            dialogue_turns = []
            for u_text, z_text in dialogue:
                total_turns += 1
                combined = (u_text + " " + z_text).lower()
                for bad_word in FORBIDDEN_KEYWORDS:
                    if bad_word in combined:
                        raise ValueError(f"Forbidden term '{bad_word}' found in {cat_name} dialogue #{d_idx}!")
                dialogue_turns.append(f"User: {u_text.strip()}\nZexo: {z_text.strip()}")
            formatted_blocks.append("\n\n".join(dialogue_turns))

    full_content = "\n\n".join(formatted_blocks) + "\n"
    OUTPUT_FILE.write_text(full_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f" Successfully generated: {OUTPUT_FILE}")
    print(f" Total Dialogues:     {total_dialogues}")
    print(f" Total Turns (U & Z): {total_turns * 2}")
    print(f" Total Lines:         {len(full_content.splitlines()):,}")
    print(f" Total Words:         {len(full_content.split()):,}")
    print(f" Total Characters:    {len(full_content):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
