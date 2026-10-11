"""Offline preference optimization for Zexo (Direct Preference Optimization).

This module implements the DPO objective over explicit chosen/rejected response
pairs. It deliberately does not treat a model's unreviewed self-generated text
as a reward: preferences should come from human review or a separately
validated evaluator. The implementation is a small CPU/NumPy reference trainer
built on the doraneural Tensor autograd API and can optimize all trainable
weights or compatible LoRA adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from doraneural.autograd import Tensor, no_grad
from doraneural.transformer import TensorAdamW


@dataclass(frozen=True)
class PreferenceExample:
    """A prompt and two candidate assistant responses, ordered by preference."""

    prompt: str
    chosen: str
    rejected: str
    example_id: str = ""


@dataclass(frozen=True)
class _TokenPreference:
    """Tokenized and context-window-bounded preference record."""

    prompt_ids: Tuple[int, ...]
    chosen_ids: Tuple[int, ...]
    rejected_ids: Tuple[int, ...]
    example_id: str


def _required_text(record: Dict[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: '{field}' must be a non-empty string")
    return value.strip()


def read_preferences(path: Union[str, Path]) -> List[PreferenceExample]:
    """Read validated preference pairs from JSONL.

    Each non-comment line has the shape::

        {"prompt": "...", "chosen": "...", "rejected": "..."}

    Additional keys are ignored by the trainer. Identical chosen/rejected
    answers are rejected because they
    contain no preference signal.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Preference data not found: {source}")

    examples: List[PreferenceExample] = []
    seen_ids = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: each record must be a JSON object")

            prompt = _required_text(record, "prompt", line_number)
            chosen = _required_text(record, "chosen", line_number)
            rejected = _required_text(record, "rejected", line_number)
            if chosen == rejected:
                raise ValueError(f"{source}:{line_number}: chosen and rejected responses are identical")
            example_id = record.get("id", f"line-{line_number}")
            if not isinstance(example_id, (str, int)):
                raise ValueError(f"{source}:{line_number}: 'id' must be a string or integer")
            example_id = str(example_id)
            if example_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate preference id {example_id!r}")
            seen_ids.add(example_id)
            examples.append(PreferenceExample(prompt, chosen, rejected, example_id))

    if not examples:
        raise ValueError(f"No preference pairs found in {source}")
    return examples


def format_preference_prompt(prompt: str, system_prompt: str) -> str:
    """Render plain prompts using the same simple role labels as Zexo chat.

    Prompts already containing a user role are kept intact and given an
    assistant prefix if needed. This lets users supply multi-turn context
    verbatim while making the one-field JSONL format convenient for new data.
    """

    text = prompt.strip()
    if not text:
        raise ValueError("preference prompt must not be empty")
    if re.search(r"(?:^|\n)(?:Zexo|Assistant):\s*$", text, re.IGNORECASE):
        return text
    if re.search(r"(?:^|\n)(?:User|Human):", text, re.IGNORECASE):
        if not re.search(r"(?:^|\n)System:", text, re.IGNORECASE) and system_prompt.strip():
            text = f"System: {system_prompt.strip()}\n{text}"
        return f"{text.rstrip()}\nZexo:"
    if system_prompt.strip():
        return f"System: {system_prompt.strip()}\nUser: {text}\nZexo:"
    return f"User: {text}\nZexo:"


def _softplus(value: Tensor) -> Tensor:
    """Numerically stable differentiable softplus for a scalar Tensor."""

    if value.data.size != 1:
        raise ValueError("DPO softplus expects a scalar")
    x = float(value.data.reshape(-1)[0])
    data = np.asarray(np.logaddexp(0.0, x), dtype=value.dtype)
    result = Tensor(data, requires_grad=value.requires_grad, dtype=value.dtype, _children=(value,), _op="softplus")
    if result.requires_grad:
        # sigmoid(x), evaluated without overflow for large positive or negative x.
        if x >= 0.0:
            derivative = 1.0 / (1.0 + math.exp(-min(x, 88.0)))
        else:
            exp_x = math.exp(max(x, -88.0))
            derivative = exp_x / (1.0 + exp_x)

        def _backward() -> None:
            grad = np.asarray(result.grad, dtype=value.dtype) * derivative
            value.grad = grad if value.grad is None else value.grad + grad

        result._backward = _backward
    return result


def _log_probability(
    model: Any,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
) -> Tensor:
    """Differentiable sum of autoregressive log probabilities for one answer."""

    if not prompt_ids:
        raise ValueError("tokenized prompt must contain at least one token")
    if not response_ids:
        raise ValueError("tokenized response must contain at least one token")
    full_ids = list(prompt_ids) + list(response_ids)
    input_ids = full_ids[:-1]
    response_start = len(prompt_ids) - 1
    logits = model.forward(input_ids)

    # Stable log-softmax. The row maximum is a constant shift; omitting its
    # derivative is exact because it cancels between logsumexp and the target.
    row_max = Tensor(np.max(logits.data, axis=-1, keepdims=True), dtype=logits.dtype)
    log_normalizer = (logits - row_max).exp().sum(axis=-1, keepdims=True).log() + row_max
    positions = np.arange(response_start, response_start + len(response_ids), dtype=np.int64)
    target_ids = np.asarray(response_ids, dtype=np.int64)
    selected = logits[positions, target_ids]
    selected_normalizer = log_normalizer.reshape(-1)[positions]
    return (selected - selected_normalizer).sum()


def _tokenize_preferences(
    examples: Sequence[PreferenceExample],
    tokenizer: Any,
    system_prompt: str,
    max_seq_len: int,
    eos_token_id: int,
) -> List[_TokenPreference]:
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2 to score a response")

    tokenized: List[_TokenPreference] = []
    prompt_reserve = max(1, min(32, max_seq_len // 4))
    max_response_tokens = max_seq_len - prompt_reserve
    for example in examples:
        formatted_prompt = format_preference_prompt(example.prompt, system_prompt)
        prompt_ids = list(tokenizer.encode(formatted_prompt, bos=True))
        if not prompt_ids:
            raise ValueError(f"Preference {example.example_id!r} produced an empty prompt")

        def encode_response(text: str) -> List[int]:
            response_ids = list(tokenizer.encode(text.strip(), bos=False))
            if not response_ids:
                raise ValueError(f"Preference {example.example_id!r} produced an empty response")
            if response_ids[-1] != eos_token_id:
                response_ids.append(eos_token_id)
            # Reserve context for the prompt so long completions cannot erase
            # the user request. Chosen/rejected share the same context budget.
            response_ids = response_ids[:max_response_tokens]
            return response_ids

        chosen_ids = encode_response(example.chosen)
        rejected_ids = encode_response(example.rejected)
        prompt_capacity = max_seq_len - max(len(chosen_ids), len(rejected_ids))
        prompt_ids = prompt_ids[-prompt_capacity:]
        if not prompt_ids:
            raise ValueError(f"Preference {example.example_id!r} has no prompt context after truncation")

        tokenized.append(
            _TokenPreference(
                tuple(prompt_ids), tuple(chosen_ids), tuple(rejected_ids), example.example_id,
            )
        )
    return tokenized


def _architecture_signature(model: Any) -> Tuple[Any, ...]:
    custom_signature = getattr(model, "architecture_signature", None)
    if custom_signature is not None:
        return tuple(custom_signature)
    return (
        model.dim,
        model.hidden_dim,
        model.n_layers,
        model.n_heads,
        model.n_kv_heads,
        model.vocab_size,
        model.seq_len,
        model.rope_type,
        model.rope_theta,
    )


class PreferenceTrainer:
    """Optimize an autoregressive policy against fixed reference preferences.

    ``reference_model`` must be a frozen copy of the policy before refinement.
    The DPO objective is ``softplus(-beta * (policy_margin - reference_margin))``.
    A validation preference file can select and restore the best epoch by
    held-out DPO loss instead of blindly keeping the last update.
    """

    def __init__(
        self,
        policy_model: Any,
        reference_model: Any,
        tokenizer: Any,
        system_prompt: str,
        beta: float = 0.1,
        lr: float = 5e-6,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        max_seq_len: Optional[int] = None,
    ) -> None:
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"beta must be finite and positive, got {beta}")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"lr must be finite and positive, got {lr}")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError(f"weight_decay must be finite and non-negative, got {weight_decay}")
        if not math.isfinite(grad_clip) or grad_clip < 0.0:
            raise ValueError(f"grad_clip must be finite and non-negative, got {grad_clip}")
        if _architecture_signature(policy_model) != _architecture_signature(reference_model):
            raise ValueError("policy and reference models must have identical architectures")
        if policy_model is reference_model:
            raise ValueError("policy and reference must be independent model copies")
        policy_parameter_ids = {id(parameter) for parameter in policy_model.parameters()}
        reference_parameter_ids = {id(parameter) for parameter in reference_model.parameters()}
        if policy_parameter_ids & reference_parameter_ids:
            raise ValueError("policy and reference must not share parameter objects")
        for parameter in reference_model.parameters():
            parameter.requires_grad = False

        self.policy = policy_model
        self.reference = reference_model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.beta = float(beta)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        requested_seq_len = policy_model.seq_len if max_seq_len is None else int(max_seq_len)
        if requested_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        self.max_seq_len = min(requested_seq_len, policy_model.seq_len)
        source_llm = getattr(policy_model, "_llm", None)
        source_config = getattr(source_llm, "config", None)
        configured_eos = getattr(policy_model, "eos_token_id", None)
        if configured_eos is None:
            configured_eos = getattr(source_config, "eos_token_id", 2)
        self.eos_token_id = int(configured_eos)

    def _prepare(self, examples: Sequence[PreferenceExample]) -> List[_TokenPreference]:
        return _tokenize_preferences(
            examples,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
        )

    def _reference_margins(self, examples: Sequence[_TokenPreference]) -> List[float]:
        margins = []
        for example in examples:
            chosen = _log_probability(self.reference, example.prompt_ids, example.chosen_ids)
            rejected = _log_probability(self.reference, example.prompt_ids, example.rejected_ids)
            margins.append(float(chosen.item() - rejected.item()))
        return margins

    def _evaluate(
        self,
        examples: Sequence[_TokenPreference],
        reference_margins: Sequence[float],
    ) -> Dict[str, float]:
        losses: List[float] = []
        correct = 0
        with no_grad():
            for example, reference_margin in zip(examples, reference_margins):
                chosen = _log_probability(self.policy, example.prompt_ids, example.chosen_ids)
                rejected = _log_probability(self.policy, example.prompt_ids, example.rejected_ids)
                policy_margin = float(chosen.item() - rejected.item())
                margin = policy_margin - reference_margin
                losses.append(float(np.logaddexp(0.0, -self.beta * margin)))
                correct += int(policy_margin > 0.0)
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": correct / len(examples) if examples else float("nan"),
        }

    def _trainable_parameters(self):
        if getattr(self.policy, "lora_config", None) is not None:
            return self.policy.lora_parameters()
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("The policy has no trainable parameters")
        return parameters

    def _clip_gradients(self, parameters: Sequence[Any]) -> float:
        norm_sq = sum(float(np.sum(p.grad * p.grad)) for p in parameters if p.grad is not None)
        norm = math.sqrt(norm_sq)
        if self.grad_clip > 0.0 and norm > self.grad_clip:
            scale = self.grad_clip / max(norm, 1e-12)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad *= scale
        return norm

    def fit(
        self,
        train_examples: Sequence[PreferenceExample],
        eval_examples: Optional[Sequence[PreferenceExample]] = None,
        epochs: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train for a small number of epochs and restore the best held-out epoch."""

        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if not train_examples:
            raise ValueError("train_examples must not be empty")
        if eval_examples is not None and not eval_examples:
            raise ValueError("eval_examples must not be empty when provided")

        train_pairs = self._prepare(train_examples)
        eval_pairs = self._prepare(eval_examples) if eval_examples is not None else None
        train_reference = self._reference_margins(train_pairs)
        eval_reference = self._reference_margins(eval_pairs) if eval_pairs is not None else None
        parameters = self._trainable_parameters()
        optimizer = TensorAdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(seed)

        history: Dict[str, Any] = {
            "loss": [],
            "preference_accuracy": [],
            "examples": len(train_pairs),
            "beta": self.beta,
            "max_seq_len": self.max_seq_len,
        }
        if eval_pairs is not None:
            history["eval_loss"] = []
            history["eval_accuracy"] = []
            best_eval_loss = float("inf")
            best_parameters: Optional[List[np.ndarray]] = None
            best_epoch = 0
        else:
            best_eval_loss = float("inf")
            best_parameters = None
            best_epoch = 0

        for epoch in range(epochs):
            order = np.arange(len(train_pairs))
            if shuffle:
                rng.shuffle(order)
            losses: List[float] = []
            correct = 0

            for index in order:
                example = train_pairs[int(index)]
                reference_margin = train_reference[int(index)]
                optimizer.zero_grad()
                chosen_logp = _log_probability(self.policy, example.prompt_ids, example.chosen_ids)
                rejected_logp = _log_probability(self.policy, example.prompt_ids, example.rejected_ids)
                margin = chosen_logp - rejected_logp - reference_margin
                scaled_margin = margin * self.beta
                loss = _softplus(-scaled_margin)
                loss.backward()
                self._clip_gradients(parameters)
                optimizer.step()

                value = float(loss.item())
                losses.append(value)
                correct += int(float(chosen_logp.item() - rejected_logp.item()) > 0.0)
                del loss, chosen_logp, rejected_logp, margin, scaled_margin
                detach_metrics = getattr(self.policy, "detach_metrics", None)
                if callable(detach_metrics):
                    detach_metrics()

            history["loss"].append(float(np.mean(losses)))
            history["preference_accuracy"].append(correct / len(train_pairs))
            if eval_pairs is not None and eval_reference is not None:
                metrics = self._evaluate(eval_pairs, eval_reference)
                history["eval_loss"].append(metrics["loss"])
                history["eval_accuracy"].append(metrics["accuracy"])
                if metrics["loss"] < best_eval_loss:
                    best_eval_loss = metrics["loss"]
                    best_parameters = [parameter.data.copy() for parameter in parameters]
                    best_epoch = epoch + 1

            if verbose:
                message = (
                    f"[DPO] Epoch {epoch + 1}/{epochs} loss={history['loss'][-1]:.4f} "
                    f"preference_acc={history['preference_accuracy'][-1]:.3f}"
                )
                if eval_pairs is not None:
                    message += (
                        f" eval_loss={history['eval_loss'][-1]:.4f}"
                        f" eval_acc={history['eval_accuracy'][-1]:.3f}"
                    )
                print(message, flush=True)

        if best_parameters is not None:
            for parameter, values in zip(parameters, best_parameters):
                parameter.data[...] = values
            history["best_epoch"] = best_epoch
            history["best_eval_loss"] = best_eval_loss
        self.policy.zero_grad()
        return history


__all__ = [
    "PreferenceExample",
    "PreferenceTrainer",
    "format_preference_prompt",
    "read_preferences",
]
