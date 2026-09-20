#!/usr/bin/env python3
"""Interactive Terminal Chat Interface for Dora-X2.

Demonstrates real-time multi-turn conversation with the 16-layer,
768-dimension, 12-head Multi-Head RTU (MH-RTU) language model.
"""

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.dora_x2 import DoraX2Config, DoraX2LM, DoraX2ChatSession


def print_banner(config: DoraX2Config):
    print("\n" + "═" * 76)
    print(" ⚡ DORA-X2 CONVERSATIONAL AI TERMINAL (v2.0)")
    print(f"    Architecture: {config.n_layers} Layers | {config.dim} Dim | {config.n_heads} Heads (MH-RTU)")
    print(f"    Parameters:   {config.parameter_count:,} ({config.parameter_count/1e6:.1f}M params)")
    print(f"    Dynamics:     Decoupled Recurrent Trace + Bio-Reflective KANs")
    print("═" * 76)
    print(" Type your message and press Enter to chat.")
    print(" Commands: /clear (reset memory), /stats (context usage), /quit (exit)")
    print("─" * 76 + "\n")


def interactive_chat_loop(
    session: DoraX2ChatSession,
    temperature: float = 0.7,
    max_tokens: int = 120,
):
    print_banner(session.model.config)

    while True:
        try:
            user_input = input("\033[1;36mUser > \033[0m").strip()
            if not user_input:
                continue

            # Command handling
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "exit", "quit"):
                print("\n👋 Goodbye from Dora-X2! Have a productive session!\n")
                break

            if cmd == "/clear":
                session.reset()
                print("\033[33m🧹 Recurrent state memory & conversation history cleared.\033[0m\n")
                continue

            if cmd == "/stats":
                p_count = session.model.config.parameter_count
                print(f"\033[33m📊 Model Specs: {session.model.config.n_layers} Layers, {session.model.config.dim} Dim, {session.model.config.n_heads} Heads\033[0m")
                print(f"\033[33m   Total Parameters: {p_count:,} ({p_count/1e6:.1f}M)\033[0m")
                print(f"\033[33m   Dialogue Turns:   {session.total_turns}\033[0m")
                print(f"\033[33m   Tokens Generated: {session.total_tokens_generated}\033[0m\n")
                continue

            if cmd == "/help":
                print("\033[33m💡 Available Commands:\033[0m")
                print("\033[33m   /clear  - Reset dialogue history and clear recurrent traces\033[0m")
                print("\033[33m   /stats  - Display model architecture and session metrics\033[0m")
                print("\033[33m   /quit   - Exit the conversational session\033[0m\n")
                continue

            # Stream generation
            sys.stdout.write("\033[1;35mDora-X2 > \033[0m")
            sys.stdout.flush()

            t0 = time.perf_counter()
            tokens_streamed = 0
            for piece in session.chat(user_input, max_tokens=max_tokens, temperature=temperature, stream=True):
                sys.stdout.write(piece)
                sys.stdout.flush()
                tokens_streamed += 1

            dt = time.perf_counter() - t0
            tps = tokens_streamed / dt if dt > 0 else 0.0
            print(f"\n\033[2m[{tokens_streamed} tokens in {dt*1000:.1f}ms | {tps:.1f} tok/s]\033[0m\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Session ended.\n")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat Terminal with Dora-X2 (16 Layers, 768 Dim, 12 Heads MH-RTU).")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pre-trained checkpoint (.npz / .json)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum generated tokens per turn")
    args = parser.parse_args()

    if args.checkpoint and Path(args.checkpoint).with_suffix(".npz").exists():
        print(f"📦 Loading checkpoint from {args.checkpoint}...")
        model = DoraX2LM.load(args.checkpoint)
    else:
        print("⚡ Initializing Dora-X2 Model (16 Layers, 768 Dim, 12 Heads MH-RTU)...")
        config = DoraX2Config()
        model = DoraX2LM(config)

    session = DoraX2ChatSession(model)
    interactive_chat_loop(session, temperature=args.temp, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
