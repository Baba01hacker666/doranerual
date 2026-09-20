"""Training Engine for Dora-X2 16-Layer MH-RTU Language Model.

Executes high-throughput sequence training on curated conversational datasets
with AdamW updates, O(1) recurrent trace state management, and real-time
evaluation benchmarking.

Supports both PyTorch GPU/CUDA acceleration and pure NumPy CPU fallback.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Force line buffering for CI/CD and Colab live output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.dora_x2 import DoraX2Config, DoraX2LM, DoraX2ChatSession

# Optional PyTorch Acceleration
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_text_data(data_path: Path, max_bytes: int = -1) -> bytes:
    """Load training text as raw bytes."""
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    raw = data_path.read_bytes()
    if max_bytes > 0 and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw


# ==============================================================================
# PyTorch Native GPU Model & Training Engine
# ==============================================================================
if HAS_TORCH:

    class PyTorchRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return (x / rms) * self.weight


    class PyTorchMultiHeadRTU(nn.Module):
        def __init__(self, dim: int, n_heads: int):
            super().__init__()
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            scale = 1.0 / math.sqrt(dim)
            head_scale = 1.0 / math.sqrt(self.head_dim)

            self.rms = PyTorchRMSNorm(dim)
            self.wq = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * scale)
            self.wk = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * head_scale)
            self.wv = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * head_scale)
            self.wg = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * scale)
            self.wo = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * scale)
            self.w_decay = nn.Parameter(torch.full((n_heads, self.head_dim), 2.5, dtype=torch.float32))

        def forward_sequence(
            self,
            x_seq: torch.Tensor,
            init_state: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, T, D = x_seq.shape
            norm_x = self.rms(x_seq)

            q = (norm_x @ self.wq).view(B, T, self.n_heads, self.head_dim)
            k = (norm_x @ self.wk).view(B, T, self.n_heads, self.head_dim)
            v = (norm_x @ self.wv).view(B, T, self.n_heads, self.head_dim)
            g = F.silu((norm_x @ self.wg).view(B, T, self.n_heads, self.head_dim))

            decay = torch.sigmoid(self.w_decay).view(1, self.n_heads, self.head_dim)

            state = torch.zeros(B, self.n_heads, self.head_dim, device=x_seq.device, dtype=x_seq.dtype) if init_state is None else init_state

            out_steps = []
            for t in range(T):
                state = decay * state + (k[:, t] * v[:, t])
                h_out = q[:, t] * state * g[:, t]
                out_steps.append(h_out)

            stacked = torch.stack(out_steps, dim=1).reshape(B, T, D)
            out = stacked @ self.wo
            return out, state


    class PyTorchDoraX2Block(nn.Module):
        def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_layers: int):
            super().__init__()
            self.dim = dim
            self.hidden_dim = hidden_dim
            self.layer_scale = 1.0 / math.sqrt(2.0 * n_layers)

            self.rtu = PyTorchMultiHeadRTU(dim, n_heads)

            scale = 1.0 / math.sqrt(dim)
            hidden_scale = 1.0 / math.sqrt(hidden_dim)
            self.rms_ffn = PyTorchRMSNorm(dim)
            self.w1 = nn.Parameter(torch.randn(dim, hidden_dim, dtype=torch.float32) * scale)
            self.w2 = nn.Parameter(torch.randn(hidden_dim, dim, dtype=torch.float32) * hidden_scale)
            self.w3 = nn.Parameter(torch.randn(dim, hidden_dim, dtype=torch.float32) * scale)

        def forward_sequence(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            rtu_out, next_state = self.rtu.forward_sequence(x, state)
            x = x + rtu_out * self.layer_scale

            norm_ffn = self.rms_ffn(x)
            gate = norm_ffn @ self.w1
            up = norm_ffn @ self.w3
            hidden = F.silu(gate) * up
            down = hidden @ self.w2

            x = x + down * self.layer_scale
            return x, next_state


    class PyTorchDoraX2LM(nn.Module):
        def __init__(self, config: DoraX2Config):
            super().__init__()
            self.config = config
            scale = 1.0 / math.sqrt(config.dim)

            self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
            nn.init.normal_(self.tok_emb.weight, std=scale)

            self.layers = nn.ModuleList([
                PyTorchDoraX2Block(config.dim, config.hidden_dim, config.n_heads, config.n_layers)
                for _ in range(config.n_layers)
            ])
            self.rms_final = PyTorchRMSNorm(config.dim)
            self.lm_head = nn.Parameter(torch.randn(config.dim, config.vocab_size, dtype=torch.float32) * scale)

        def forward(
            self,
            input_ids: torch.Tensor,
            states: Optional[List[torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            B, T = input_ids.shape
            x = self.tok_emb(input_ids)

            if states is None:
                states = [None] * len(self.layers)

            next_states = []
            for l, layer in enumerate(self.layers):
                x, st = layer.forward_sequence(x, states[l])
                next_states.append(st)

            norm_final = self.rms_final(x)
            logits = norm_final @ self.lm_head
            return logits, next_states

        def save_checkpoint(self, path: Union[str, Path]) -> Path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            manifest = asdict(self.config)
            manifest["parameter_count"] = self.config.parameter_count
            p.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            tensors: Dict[str, np.ndarray] = {
                "tok_emb": self.tok_emb.weight.detach().cpu().float().numpy(),
                "rms_final": self.rms_final.weight.detach().cpu().float().numpy(),
                "lm_head": self.lm_head.detach().cpu().float().numpy(),
            }
            for l, layer in enumerate(self.layers):
                tensors[f"layer_{l}_rtu_rms"] = layer.rtu.rms.weight.detach().cpu().float().numpy()
                tensors[f"layer_{l}_rtu_wq"] = layer.rtu.wq.detach().cpu().float().numpy()
                tensors[f"layer_{l}_rtu_wk"] = layer.rtu.wk.detach().cpu().float().numpy()
                tensors[f"layer_{l}_rtu_wv"] = layer.rtu.wv.detach().cpu().float().numpy()
                tensors[f"layer_{l}_rtu_wg"] = layer.rtu.wg.detach().cpu().float().numpy()
                tensors[f"layer_{l}_rtu_wo"] = layer.rtu.wo.detach().cpu().float().numpy()
                tensors[f"layer_{l}_rtu_decay"] = layer.rtu.w_decay.detach().cpu().float().numpy()

                tensors[f"layer_{l}_ffn_rms"] = layer.rms_ffn.weight.detach().cpu().float().numpy()
                tensors[f"layer_{l}_ffn_w1"] = layer.w1.detach().cpu().float().numpy()
                tensors[f"layer_{l}_ffn_w2"] = layer.w2.detach().cpu().float().numpy()
                tensors[f"layer_{l}_ffn_w3"] = layer.w3.detach().cpu().float().numpy()

            npz_path = p.with_suffix(".npz")
            np.savez(npz_path, **tensors)
            return npz_path


def train_dora_x2_pytorch(
    data_path: Path,
    output_dir: Path,
    dim: int = 768,
    hidden_dim: int = 2048,
    n_layers: int = 16,
    n_heads: int = 12,
    epochs: int = 3,
    chunk_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    max_bytes: int = 200000,
    tag: str = "dora_x2_latest",
    test_prompt: str = "User: Hello Dora-X2! What is your architecture?\nAssistant: ",
    device_name: str = "auto",
) -> Dict[str, any]:
    # Select Device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_cuda = device.type == "cuda"
    device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if use_cuda else "CPU"

    print("=" * 80)
    print(f" ⚡ DORA-X2 16-LAYER MH-RTU PYTORCH ACCELERATION [{device_label}]")
    print("=" * 80)
    print(f"Run Tag:            {tag}")
    print(f"Architecture:       {n_layers} Layers | {dim} Dim | {n_heads} Heads (head_dim={dim//n_heads})")
    print(f"FFN Hidden Dim:     {hidden_dim} (Standard SwiGLU)")
    print(f"Corpus Path:        {data_path} (max_bytes={max_bytes})")
    print(f"Hyperparameters:    epochs={epochs}, chunk_size={chunk_size}, lr={lr}")
    print(f"Hardware Engine:    PyTorch {torch.__version__} ({device_label})")
    print(f"Output Directory:   {output_dir}")
    print("-" * 80, flush=True)

    # 1. Load Data
    raw_bytes = load_text_data(data_path, max_bytes=max_bytes)
    print(f"Loaded {len(raw_bytes):,} raw bytes of training data.", flush=True)

    # 2. Build Model
    config = DoraX2Config(
        name="dora-x2",
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=256,
    )
    model = PyTorchDoraX2LM(config).to(device)
    print(f"Model instantiated with {config.parameter_count:,} parameters ({config.parameter_count/1e6:.1f}M).", flush=True)
    print("-" * 80, flush=True)

    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # 4. Pre-Training Generation Probe (using NumPy reference loader)
    temp_probe_path = output_dir / "_probe_init"
    model.save_checkpoint(temp_probe_path)
    probe_lm = DoraX2LM.load(temp_probe_path)
    print(f"🔍 Pre-Training Generation Probe (Prompt: '{test_prompt[:40]}...'):")
    pre_gen = probe_lm.generate(test_prompt, max_tokens=35, temperature=0.7)
    print(f"  ➜ Generated: \"{pre_gen.strip()}\"", flush=True)
    print("-" * 80, flush=True)

    # 5. Training Loop
    t_start = time.perf_counter()
    n_chunks = max(1, (len(raw_bytes) - 1) // chunk_size)
    print(f"🚀 Commencing PyTorch training across {epochs} epochs ({n_chunks} chunks of {chunk_size} bytes per epoch)...", flush=True)

    loss_history = []
    total_tokens_trained = 0

    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        ep_tokens = 0
        ep_start = time.perf_counter()
        states = None

        for c_idx in range(n_chunks):
            start = c_idx * chunk_size
            end = min(len(raw_bytes), start + chunk_size + 1)
            chunk_bytes = raw_bytes[start:end]
            if len(chunk_bytes) < 2:
                continue

            # Tensor chunk
            tok_tensor = torch.tensor(list(chunk_bytes), dtype=torch.long, device=device).unsqueeze(0)
            in_ids = tok_tensor[:, :-1]
            tgt_ids = tok_tensor[:, 1:]
            n_toks = in_ids.size(1)

            if c_idx % 4 == 0:
                states = None
            elif states is not None:
                states = [s.detach() if s is not None else None for s in states]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits, states = model(in_ids, states=states)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), tgt_ids.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            ep_loss += loss_val * n_toks
            ep_tokens += n_toks
            total_tokens_trained += n_toks

            if (c_idx + 1) % 25 == 0 or (c_idx + 1) == n_chunks:
                elapsed = time.perf_counter() - ep_start
                cur_loss = ep_loss / max(1, ep_tokens)
                tps = ep_tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"   [Epoch {ep:02d}/{epochs:02d}] Step {c_idx+1:04d}/{n_chunks:04d} "
                    f"| Loss: {cur_loss:.4f} | Speed: {tps:.1f} tok/s",
                    flush=True,
                )

        avg_loss = ep_loss / max(1, ep_tokens)
        loss_history.append(avg_loss)
        ep_time = time.perf_counter() - ep_start
        print(f"✨ Epoch {ep:02d}/{epochs:02d} Complete -> Avg Loss: {avg_loss:.4f} in {ep_time:.1f}s", flush=True)

    total_time = time.perf_counter() - t_start
    print("-" * 80, flush=True)
    print(f"✓ Dora-X2 Training finished in {total_time:.2f}s ({total_tokens_trained / total_time:.1f} tok/s)", flush=True)
    print("-" * 80, flush=True)

    # 6. Save Checkpoint (identical .npz and .json format compatible with DoraX2LM)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest"
    tagged_path = output_dir / tag
    model.save_checkpoint(latest_path)
    model.save_checkpoint(tagged_path)
    print("💾 Dora-X2 Checkpoint Saved:")
    print(f"  • Latest: {latest_path.with_suffix('.npz')}")
    print(f"  • Tagged: {tagged_path.with_suffix('.npz')}")
    print(f"  • Schema: {latest_path.with_suffix('.json')}")
    print("=" * 80, flush=True)

    # 7. Post-Training Generation Probe
    post_lm = DoraX2LM.load(latest_path)
    print(f"✨ Post-Training Generation Probe (Prompt: '{test_prompt[:40]}...'):")
    post_gen = post_lm.generate(test_prompt, max_tokens=40, temperature=0.7)
    print(f"  ➜ Generated: \"{post_gen.strip()}\"", flush=True)
    print("=" * 80, flush=True)

    return {
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "duration_s": total_time,
        "tokens_processed": total_tokens_trained,
    }


def train_dora_x2_numpy(
    data_path: Path,
    output_dir: Path,
    dim: int = 768,
    hidden_dim: int = 2048,
    n_layers: int = 16,
    n_heads: int = 12,
    epochs: int = 3,
    chunk_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    max_bytes: int = 200000,
    tag: str = "dora_x2_latest",
    test_prompt: str = "User: Hello Dora-X2! What is your architecture?\nAssistant: ",
) -> Dict[str, any]:
    print("=" * 80)
    print(" ⚡ DORA-X2 16-LAYER MH-RTU CONVERSATIONAL TRAINING (NUMPY ENGINE)")
    print("=" * 80)
    print(f"Run Tag:            {tag}")
    print(f"Architecture:       {n_layers} Layers | {dim} Dim | {n_heads} Heads (head_dim={dim//n_heads})")
    print(f"FFN Hidden Dim:     {hidden_dim} (Standard SwiGLU)")
    print(f"Corpus Path:        {data_path} (max_bytes={max_bytes})")
    print(f"Hyperparameters:    epochs={epochs}, chunk_size={chunk_size}, lr={lr}")
    print(f"Output Directory:   {output_dir}")
    print("-" * 80, flush=True)

    raw_bytes = load_text_data(data_path, max_bytes=max_bytes)
    print(f"Loaded {len(raw_bytes):,} raw bytes of training data.", flush=True)

    config = DoraX2Config(
        name="dora-x2",
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=256,
    )
    model = DoraX2LM(config)
    print(f"Model instantiated with {config.parameter_count:,} parameters ({config.parameter_count/1e6:.1f}M).", flush=True)
    print("-" * 80, flush=True)

    print(f"🔍 Pre-Training Generation Probe (Prompt: '{test_prompt[:40]}...'):")
    pre_gen = model.generate(test_prompt, max_tokens=35, temperature=0.7)
    print(f"  ➜ Generated: \"{pre_gen.strip()}\"", flush=True)
    print("-" * 80, flush=True)

    t_start = time.perf_counter()
    n_chunks = max(1, (len(raw_bytes) - 1) // chunk_size)
    print(f"🚀 Commencing training across {epochs} epochs ({n_chunks} chunks of {chunk_size} bytes per epoch)...", flush=True)

    loss_history = []
    ep_tokens = 0
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        ep_tokens = 0
        ep_start = time.perf_counter()

        for c_idx in range(n_chunks):
            start = c_idx * chunk_size
            end = min(len(raw_bytes), start + chunk_size + 1)
            chunk = list(raw_bytes[start:end])
            if len(chunk) < 2:
                continue

            res = model.train_sequence(chunk, lr=lr, weight_decay=weight_decay, reset_state=(c_idx % 4 == 0))
            ep_loss += res["loss"] * res["tokens"]
            ep_tokens += res["tokens"]

            if (c_idx + 1) % 25 == 0 or (c_idx + 1) == n_chunks:
                elapsed = time.perf_counter() - ep_start
                cur_loss = ep_loss / max(1, ep_tokens)
                tps = ep_tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"   [Epoch {ep:02d}/{epochs:02d}] Step {c_idx+1:04d}/{n_chunks:04d} "
                    f"| Loss: {cur_loss:.4f} | Speed: {tps:.1f} tok/s",
                    flush=True,
                )

        avg_loss = ep_loss / max(1, ep_tokens)
        loss_history.append(avg_loss)
        ep_time = time.perf_counter() - ep_start
        print(f"✨ Epoch {ep:02d}/{epochs:02d} Complete -> Avg Loss: {avg_loss:.4f} in {ep_time:.1f}s", flush=True)

    total_time = time.perf_counter() - t_start
    print("-" * 80, flush=True)
    print(f"✓ Dora-X2 Training finished in {total_time:.2f}s ({ep_tokens * epochs / total_time:.1f} tok/s)", flush=True)
    print("-" * 80, flush=True)

    print(f"✨ Post-Training Generation Probe (Prompt: '{test_prompt[:40]}...'):")
    post_gen = model.generate(test_prompt, max_tokens=40, temperature=0.7)
    print(f"  ➜ Generated: \"{post_gen.strip()}\"", flush=True)
    print("-" * 80, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest"
    tagged_path = output_dir / tag
    model.save(latest_path)
    model.save(tagged_path)
    print("💾 Dora-X2 Checkpoint Saved:")
    print(f"  • Latest: {latest_path.with_suffix('.npz')}")
    print(f"  • Tagged: {tagged_path.with_suffix('.npz')}")
    print(f"  • Schema: {latest_path.with_suffix('.json')}")
    print("=" * 80, flush=True)

    return {
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "duration_s": total_time,
        "tokens_processed": ep_tokens * epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dora-X2 16-Layer MH-RTU Language Model.")
    default_data = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"
    if not default_data.exists():
        default_data = REPO_ROOT / "zexo" / "data" / "step1_conversational_expanded.txt"

    parser.add_argument("--data", type=Path, default=default_data, help="Path to training text file.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "dora_x2", help="Directory to save checkpoints.")
    parser.add_argument("--dim", type=int, default=768, help="Model representation dimension.")
    parser.add_argument("--layers", type=int, default=16, help="Number of MH-RTU layers.")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention / recurrent heads.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Recurrent chunk length.")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--max-bytes", type=int, default=100000, help="Max corpus bytes to process (-1 for full).")
    parser.add_argument("--tag", type=str, default="dora_x2_run_1", help="Checkpoint tag.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "numpy"],
        help="Training hardware device: auto, cuda (GPU), cpu, or numpy.",
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="User: Hello Dora-X2! What is your architecture?\nAssistant: ",
        help="Evaluation prompt.",
    )
    args = parser.parse_args()

    # Route to PyTorch if available, else NumPy
    if HAS_TORCH and args.device != "numpy":
        train_dora_x2_pytorch(
            data_path=args.data,
            output_dir=args.output_dir,
            dim=args.dim,
            hidden_dim=args.dim * 8 // 3,
            n_layers=args.layers,
            n_heads=args.heads,
            epochs=args.epochs,
            chunk_size=args.chunk_size,
            lr=args.lr,
            max_bytes=args.max_bytes,
            tag=args.tag,
            test_prompt=args.test_prompt,
            device_name=args.device,
        )
    else:
        train_dora_x2_numpy(
            data_path=args.data,
            output_dir=args.output_dir,
            dim=args.dim,
            hidden_dim=args.dim * 8 // 3,
            n_layers=args.layers,
            n_heads=args.heads,
            epochs=args.epochs,
            chunk_size=args.chunk_size,
            lr=args.lr,
            max_bytes=args.max_bytes,
            tag=args.tag,
            test_prompt=args.test_prompt,
        )


if __name__ == "__main__":
    main()
