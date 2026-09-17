#!/usr/bin/env python3
"""Zexo AI: Interactive Terminal Conversational Interface.

Enables real-time, streaming, multi-turn conversations with Zexo
in the terminal with slash commands, sliding context window, and graceful exit.
"""

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zexo.model import load_zexo


def print_banner(model_name: str, tier: str, params: int, backend: str):
    print("\n" + "═" * 70)
    print(f" 🤖 Welcome to Zexo AI Interactive Terminal (v1.0)")
    print(f"    Tier: {tier.upper()} | Parameters: {params:,} | Engine: {backend.upper()}")
    print("═" * 70)
    print(" Type your message and press Enter to chat.")
    print(" Commands: /clear (reset memory), /stats (show context), /quit (exit)")
    print("─" * 70 + "\n")


def interactive_chat_loop(zexo, temperature: float = 0.7, max_tokens: int = 80):
    cfg = zexo.config
    print_banner(cfg.name, cfg.tier, cfg.parameter_count, zexo.llm.backend)

    while True:
        try:
            user_input = input("\033[1;36mUser > \033[0m").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
                print("\n👋 Goodbye from Zexo! Have a great day!\n")
                break

            if user_input.lower() == "/clear":
                zexo.reset()
                print("\033[33m🧹 Conversation history and context memory cleared.\033[0m\n")
                continue

            if user_input.lower() == "/stats":
                usage = zexo.chat_session.get_context_usage()
                print(f"\033[33m📊 Context Usage: {usage['used_tokens']}/{usage['max_context_tokens']} tokens ({usage['percent_used']:.1f}%)\033[0m")
                print(f"\033[33m   Total turns: {zexo.chat_session.total_turns}, Total tokens: {zexo.chat_session.total_tokens_generated}\033[0m\n")
                continue

            if user_input.lower() == "/help":
                print("\033[33m💡 Commands: /clear, /stats, /help, /quit\033[0m\n")
                continue

            # Generate Zexo response
            sys.stdout.write("\033[1;35mZexo > \033[0m")
            sys.stdout.flush()

            t0 = time.perf_counter()
            tokens_streamed = 0
            for piece in zexo.chat(user_input, temperature=temperature, max_tokens=max_tokens, stream=True):
                sys.stdout.write(piece)
                sys.stdout.flush()
                tokens_streamed += 1
            dt = time.perf_counter() - t0
            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye from Zexo!\n")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive chat session with Zexo AI.")
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to trained Zexo checkpoint .bin file.",
    )
    parser.add_argument(
        "--tier", "-t",
        choices=["micro", "mini", "chat", "base", "large"],
        default="mini",
        help="Zexo architectural tier (default: mini).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens", "-n",
        type=int,
        default=80,
        help="Maximum tokens to generate per response turn (default: 80).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpp", "numpy"],
        default="auto",
        help="Execution engine backend (default: auto).",
    )

    args = parser.parse_args()
    zexo = load_zexo(checkpoint_path=args.checkpoint, tier=args.tier, backend=args.backend)
    interactive_chat_loop(zexo, temperature=args.temp, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
