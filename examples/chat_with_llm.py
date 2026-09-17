#!/usr/bin/env python3
"""doraneural: Interactive Conversational Chat & Sliding Context Window.

Demonstrates:
1. Programmatic multi-turn conversation with `ChatSession`
2. Automatic sliding context window eviction (older messages dropped as window fills)
3. Lookahead stop-sequence buffering (halts before hallucinating next user turns)
4. Interactive terminal chat loop with slash commands (/clear, /context, /temp, /stats)

Usage:
    # Run this example script:
    python examples/chat_with_llm.py

    # Or run interactive chat from CLI:
    doraneural chat
    doraneural chat --model stories260K --tokens 40
"""

import sys
from pathlib import Path

# Ensure doraneural in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import doraneural as dn


def demo_programmatic_chat():
    print("=" * 68)
    print("🤖 1. Programmatic Multi-Turn Chat with Context Management")
    print("=" * 68)

    # 1. Load model (arnir0/Tiny-LLM 10M params or stories260K)
    print("Loading model (arnir0/Tiny-LLM)...")
    llm = dn.load_pretrained_llm("arnir0/Tiny-LLM")

    # 2. Create chat session with custom context limit (e.g. 128 tokens for demonstration)
    session = dn.ChatSession(
        llm=llm,
        system_prompt="You are a friendly and knowledgeable AI assistant.",
        max_context_tokens=128,  # Small window to demonstrate sliding context eviction
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
    )

    questions = [
        "Hello! Who are you?",
        "What is your favorite animal?",
        "Why do birds have wings?",
        "Tell me a short secret.",
    ]

    for q in questions:
        print(f"\n👤 User: {q}")
        reply = session.chat(q)
        print(f"🤖 Assistant: {reply}")

        usage = session.get_context_usage()
        print(f"   [Context Window: {usage['used_tokens']}/{usage['max_context']} tokens ({usage['percentage']}%) | {usage['messages_count']} messages in memory]")

    print("\n" + "=" * 68)
    print("Session Stats:")
    print(f"Total turns: {session.total_turns}")
    print(f"Total tokens generated: {session.total_tokens_generated}")
    print("=" * 68)


def demo_interactive_chat():
    print("\n" + "=" * 68)
    print("💬 2. Launching Interactive Terminal Chat Session")
    print("=" * 68)

    llm = dn.load_pretrained_llm("arnir0/Tiny-LLM")
    session = dn.ChatSession(
        llm=llm,
        system_prompt="You are an intelligent, friendly AI assistant.",
        max_new_tokens=64,
        temperature=0.7,
    )
    session.interactive_loop()


if __name__ == "__main__":
    if "--interactive" in sys.argv or "-i" in sys.argv:
        demo_interactive_chat()
    else:
        demo_programmatic_chat()
        print("\n💡 Tip: Run with --interactive to start the live interactive chat REPL:")
        print("   python examples/chat_with_llm.py --interactive")
        print("   or run: doraneural chat")
