"""doraneural: Interactive Multi-Turn Chat & Context Window Manager.

Provides a full conversational chat interface for pretrained LLaMA models
(both SafeTensors HF models and llama2.c .bin weights) with:
- Multi-turn conversation history management
- Configurable sliding context window with automatic turn eviction
- Lookahead buffer for stop sequences (e.g. "User:", "Human:") to prevent hallucination
- Typewriter token streaming in real-time
- Rich terminal interactive REPL with slash commands (/clear, /context, /temp, /stats)
- Graceful Ctrl+C handling
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from .llm import LlamaLLM, load_pretrained_llm


@dataclass
class ChatMessage:
    """A single turn in a conversational history."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ChatSession:
    """Interactive multi-turn conversation session with context window management.

    Parameters:
        llm: Initialized LlamaLLM instance.
        system_prompt: Background persona or instruction.
        max_context_tokens: Hard ceiling for total tokens (prompt + output).
                            Defaults to the model's seq_len (e.g. 1024).
        max_new_tokens: Maximum tokens to generate per response turn.
        temperature: Sampling temperature (0.0 = greedy argmax, 0.7 = balanced).
        top_p: Nucleus sampling threshold.
        stop_sequences: List of strings that halt generation when encountered.
    """

    DEFAULT_STOP_SEQUENCES = [
        "\nUser:",
        "User:",
        "\nHuman:",
        "Human:",
        "\nZexo:",
        "\nSystem:",
        "System:",
        "</s>",
    ]

    def __init__(
        self,
        llm: LlamaLLM,
        system_prompt: str = "You are a helpful and concise AI assistant.",
        max_context_tokens: Optional[int] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens or llm.config.seq_len
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.stop_sequences = stop_sequences or list(self.DEFAULT_STOP_SEQUENCES)

        self.messages: List[ChatMessage] = []
        self.total_tokens_generated: int = 0
        self.total_turns: int = 0

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append(ChatMessage(role=role, content=content.strip()))

    def clear(self) -> None:
        """Reset conversation history and clear model KV cache."""
        self.messages.clear()
        self.llm.reset_cache()

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the model's tokenizer."""
        return len(self.llm.tokenizer.encode(text, bos=False))

    # Label used as the assistant prefix — must match what the model was trained on
    ASSISTANT_PREFIX: str = "Zexo"

    def format_prompt(self, pending_user_input: Optional[str] = None) -> str:
        """Format the active message history into a dialogue prompt string."""
        prefix = self.ASSISTANT_PREFIX
        lines: List[str] = []
        if self.system_prompt:
            lines.append(f"System: {self.system_prompt}")

        for msg in self.messages:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"{prefix}: {msg.content}")
            elif msg.role == "system":
                lines.append(f"System: {msg.content}")

        if pending_user_input is not None:
            lines.append(f"User: {pending_user_input.strip()}")

        lines.append(f"{prefix}:")
        return "\n".join(lines)

    def trim_history(self, pending_user_input: Optional[str] = None) -> int:
        """Slide context window by evicting oldest turns until prompt fits budget.

        Returns:
            int: Number of evicted messages.
        """
        budget = self.max_context_tokens - self.max_new_tokens
        if budget < 32:
            budget = max(16, self.max_context_tokens // 2)

        evicted = 0
        while self.messages:
            prompt = self.format_prompt(pending_user_input)
            tok_count = len(self.llm.tokenizer.encode(prompt, bos=True))
            if tok_count <= budget:
                break

            # Evict the oldest message
            self.messages.pop(0)
            evicted += 1

        return evicted

    def get_context_usage(self, pending_user_input: Optional[str] = None) -> Dict[str, Union[int, float, str]]:
        """Calculate current context window metrics."""
        prompt = self.format_prompt(pending_user_input)
        prompt_tokens = len(self.llm.tokenizer.encode(prompt, bos=True))
        max_ctx = self.max_context_tokens
        pct = (prompt_tokens / max_ctx) * 100.0 if max_ctx > 0 else 0.0

        return {
            "used_tokens": prompt_tokens,
            "max_context": max_ctx,
            "max_context_tokens": max_ctx,
            "percentage": round(pct, 1),
            "percent_used": round(pct, 1),
            "messages_count": len(self.messages),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "backend": self.llm.backend,
        }

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Send a user message, stream the assistant's reply, and save to history.

        Uses lookahead buffer to stop precisely before any stop sequence.
        """
        user_text = user_input.strip()
        if not user_text:
            return

        # 1. Slide window to ensure prompt fits context budget
        evicted = self.trim_history(pending_user_input=user_text)

        # 2. Add user message
        self.add_message("user", user_text)

        # 3. Format full dialogue prompt
        prompt = self.format_prompt()

        # 4. Generate with lookahead stop-sequence filter
        buffer = ""
        full_response_parts: List[str] = []
        stopped = False

        self.llm.reset_cache()
        for piece in self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
        ):
            buffer += piece

            # Check if any full stop sequence appears in buffer
            for stop_seq in self.stop_sequences:
                if stop_seq in buffer:
                    idx = buffer.find(stop_seq)
                    clean_to_yield = buffer[:idx]
                    if clean_to_yield:
                        full_response_parts.append(clean_to_yield)
                        yield clean_to_yield
                    buffer = ""
                    stopped = True
                    break

            if stopped:
                break

            # Lookahead check: does buffer end with a prefix of any stop sequence?
            longest_prefix_len = 0
            for stop_seq in self.stop_sequences:
                for p_len in range(1, len(stop_seq)):
                    prefix = stop_seq[:p_len]
                    if buffer.endswith(prefix):
                        longest_prefix_len = max(longest_prefix_len, p_len)

            # Yield safe part, keep potential stop-sequence prefix in buffer
            safe_to_yield = buffer[: len(buffer) - longest_prefix_len]
            if safe_to_yield:
                full_response_parts.append(safe_to_yield)
                yield safe_to_yield
                buffer = buffer[len(buffer) - longest_prefix_len :]

        # Flush any remaining buffer if not a stop sequence
        if buffer and not stopped:
            full_response_parts.append(buffer)
            yield buffer

        # 5. Record assistant response in history
        assistant_reply = "".join(full_response_parts).strip()
        if not assistant_reply:
            assistant_reply = "..."
        self.add_message("assistant", assistant_reply)

        # 6. Update stats
        resp_tokens = len(self.llm.tokenizer.encode(assistant_reply, bos=False))
        self.total_tokens_generated += resp_tokens
        self.total_turns += 1

    def chat(self, user_input: str) -> str:
        """Synchronous chat turn. Returns complete response string."""
        pieces = list(self.stream_chat(user_input))
        return "".join(pieces).strip()

    def interactive_loop(self) -> None:
        """Run terminal interactive chat REPL with slash commands and rich UI."""
        # ANSI color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        DIM = "\033[2m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        model_name = getattr(self.llm, "model_path", "LLaMA")
        if hasattr(model_name, "name"):
            model_name = model_name.name

        print("=" * 68)
        print(f"{BOLD}{CYAN}💬 doraneural: Interactive LLM Chat{RESET}")
        print("=" * 68)
        print(f"Model:    {model_name} ({self.llm.config.dim} dim, {self.llm.config.n_layers} layer)")
        print(f"Backend:  {self.llm.backend.upper()} (Zero external dependencies)")
        print(f"Context:  {self.max_context_tokens} tokens max (Automatic Sliding Window)")
        print(f"Commands: {YELLOW}/clear{RESET}, {YELLOW}/context{RESET}, {YELLOW}/temp <val>{RESET}, {YELLOW}/tokens <val>{RESET}, {YELLOW}/stats{RESET}, {YELLOW}/exit{RESET}")
        print("=" * 68)
        print(f"{DIM}System prompt: \"{self.system_prompt}\"{RESET}\n")

        while True:
            try:
                user_input = input(f"{BOLD}{GREEN}👤 You:{RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{YELLOW}Exiting chat session. Goodbye!{RESET}")
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                if cmd in ("/exit", "/quit", "/q"):
                    print(f"{YELLOW}Exiting chat. Total turns: {self.total_turns}, tokens: {self.total_tokens_generated}. Bye!{RESET}")
                    break

                elif cmd in ("/clear", "/reset"):
                    self.clear()
                    print(f"{YELLOW}🧹 Conversation history and KV cache cleared.{RESET}\n")
                    continue

                elif cmd in ("/context", "/info"):
                    usage = self.get_context_usage()
                    bar_len = 25
                    filled = int(bar_len * (usage["percentage"] / 100.0))
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\n{BOLD}Context Window Status:{RESET}")
                    print(f"  [{bar}] {usage['used_tokens']} / {usage['max_context']} tokens ({usage['percentage']}%)")
                    print(f"  Messages in memory: {usage['messages_count']}")
                    print(f"  Max response tokens: {usage['max_new_tokens']}")
                    print(f"  Sampling temperature: {usage['temperature']}")
                    print(f"  Active backend: {usage['backend'].upper()}\n")
                    continue

                elif cmd == "/temp":
                    if arg:
                        try:
                            val = float(arg)
                            self.temperature = max(0.0, min(2.0, val))
                            print(f"{YELLOW}Temperature set to {self.temperature:.2f}{RESET}\n")
                        except ValueError:
                            print(f"{YELLOW}Invalid temperature: {arg}{RESET}\n")
                    else:
                        print(f"Current temperature: {self.temperature:.2f}\n")
                    continue

                elif cmd == "/tokens":
                    if arg:
                        try:
                            val = int(arg)
                            self.max_new_tokens = max(1, min(self.max_context_tokens // 2, val))
                            print(f"{YELLOW}Max new tokens set to {self.max_new_tokens}{RESET}\n")
                        except ValueError:
                            print(f"{YELLOW}Invalid number of tokens: {arg}{RESET}\n")
                    else:
                        print(f"Current max new tokens: {self.max_new_tokens}\n")
                    continue

                elif cmd == "/system":
                    if arg:
                        self.system_prompt = arg
                        print(f"{YELLOW}System prompt updated to: \"{self.system_prompt}\"{RESET}\n")
                    else:
                        print(f"Current system prompt: \"{self.system_prompt}\"\n")
                    continue

                elif cmd == "/stats":
                    print(f"\n{BOLD}Session Statistics:{RESET}")
                    print(f"  Total conversational turns: {self.total_turns}")
                    print(f"  Total tokens generated:     {self.total_tokens_generated}")
                    print(f"  Messages in memory:         {len(self.messages)}")
                    print(f"  Max context window:         {self.max_context_tokens} tokens\n")
                    continue

                elif cmd in ("/help", "/?"):
                    print(f"\n{BOLD}Available Slash Commands:{RESET}")
                    print("  /clear, /reset   - Clear chat history and memory")
                    print("  /context         - View current token usage of the context window")
                    print("  /temp <float>    - Set sampling temperature (e.g. /temp 0.5)")
                    print("  /tokens <int>    - Set max response tokens (e.g. /tokens 80)")
                    print("  /system <text>   - Update assistant system prompt")
                    print("  /stats           - View session statistics")
                    print("  /exit, /quit     - Leave chat session\n")
                    continue

                else:
                    print(f"{YELLOW}Unknown command '{cmd}'. Type /help for a list of commands.{RESET}\n")
                    continue

            # Stream response
            print(f"{BOLD}{CYAN}🤖 Assistant:{RESET} ", end="", flush=True)
            t0 = time.perf_counter()
            token_count = 0

            try:
                for piece in self.stream_chat(user_input):
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                    token_count += 1
            except KeyboardInterrupt:
                print(f"\n{YELLOW}[Generation interrupted by user]{RESET}")

            elapsed = time.perf_counter() - t0
            speed = token_count / max(1e-4, elapsed)
            print(f"\n{DIM}({token_count} tokens in {elapsed:.2f}s = {speed:.1f} tok/s){RESET}\n")
