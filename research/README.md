# 🔬 Doraneural Research Lab: Novel Neural Architectures

Welcome to the **Doraneural Research Lab**. This folder documents the exploration, mathematical derivations, empirical benchmarks, and architectural breakthroughs created to challenge classical artificial neuron paradigms.

---

## 🧭 Master Architecture Index

| # | Architecture | Module | Core Concept / Breakthrough | Primary Benchmark | Doc Link |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | **Dendritic Pyramidal Neurons** | [`DendriticDense`](../doraneural/layers.py#L618) | Multi-compartment pyramidal branches with multiplicative gating | 1-Layer XOR & Concentric Donut | [Read Doc](01_dendritic_pyramidal_neurons.md) |
| **02** | **Kolmogorov-Arnold Network** | [`ChebyshevKAN`](../doraneural/layers.py#L797) | Continuous orthogonal Chebyshev polynomial curves on synapses | Spectral Bias Wave Fitting (18.2x lower MSE) | [Read Doc](02_chebyshev_kan.md) |
| **03** | **Novel Transformers** | [`KANTransformerBlock`](../doraneural/attention.py#L393)<br>[`DendriticTransformerBlock`](../doraneural/attention.py#L484) | KAN-Former & Dendritic-Former sequence modeling | Coupled Non-linear Sequence Dynamics | [Read Doc](03_novel_transformer_blocks.md) |
| **04** | **Threshold-Gated Bifurcation** | [`BifurcatedDense`](../doraneural/layers.py#L981) | Dynamic conditional routing between sub- and supra-threshold pathways | Dual-Regime Physics (2.0x lower MSE) | [Read Doc](04_threshold_gated_bifurcation.md) |
| **05** | **Cortical Reflection Feedback** | [`ReflectiveDense`](../doraneural/layers.py#L1101) | Iterative top-down reflection loops (predictive coding) | Ambiguous / Noisy Signal Resolution | [Read Doc](05_cortical_reflection_feedback.md) |
| **06** | **Synaptic Inverters & 4D Tensors** | [`InvertedDense`](../doraneural/layers.py#L1345)<br>[`Tensor4DDense`](../doraneural/layers.py#L1491)<br>[`Inverter`](../doraneural/activations.py#L305) | Learnable forward-inverse duality ($+W \leftrightarrow -W$) & 4D Spacetime tensors | Antagonistic Physics & 4D Wave Diffusion | [Read Doc](06_synaptic_inversion_and_4d_tensors.md) |
| **07** | **Bio-Reflective KAN** | `BioHybridNet` | Unified architecture synthesizing Reflection + Routing + Dendrites + KAN | Noisy 8x8 Digit Classification (96.4% acc) | [Read Doc](07_bio_reflective_kan_unified.md) |
| **08** | **Wave, Chaos & Tunneling** | [`ComplexWaveDense`](../doraneural/layers.py#L1583)<br>[`FractalChaosDense`](../doraneural/layers.py#L1771)<br>[`TunnelingDense`](../doraneural/layers.py#L1926) | Rotational wave interference, chaotic attractors, and quantum barrier tunneling | Wave Cancellation, Chaotic Attractors & Dead-Neuron Immunity | [Read Doc](08_complex_chaos_and_tunneling.md) |
| **09** | **LLM TPS Optimization** | Native C++ SIMD Engine | AVX-512 VNNI U8/I8 quantization, 4-row blocking, and OpenMP thread tuning | >1000 TPS on CPU Inference | [Read Doc](09_llm_tps_optimization.md) |
| **10** | **Recurrent Trace Units (RTU)** | `RTULanguageModel` (Research Sandbox) | Tokenizer-free 256-byte vocabulary, $O(1)$ state memory, forward RTRL eligibility traces & JEPA latent prediction | Zero-RAM-Growth Streaming & Representation Non-Collapse | [Read Doc](10_recurrent_trace_units_latent.md) |
| **11** | **Comparative Study: Dora vs Transformer vs RTU** | Standard Attention vs Bio-Reflective KAN vs RTU | Empirical breakdown of vocabulary tax, $O(1)$ recurrent memory, gradient pathology, and C++ engine dispatch | Loss convergence, memory scaling & generation | [Read Doc](11_comparative_study_dora_transformer_vs_rtu.md) |
| **12** | **Jev: System-1 Non-Autoregressive AI** | [`JevDecisionModel`](../doraneural/jev.py) | Non-autoregressive parallel evaluation, typed decision primitives (`Choice`, `Score`), and RLCD calibration | Sub-2ms Latency vs Multi-Second LLMs (>200x speedup) | [Read Doc](12_jev_system_one_non_autoregressive_ai.md) |

---

## 🎯 Guiding Philosophy

> *"Classical artificial neurons have remained essentially unchanged since McCulloch & Pitts (1943) and Rosenblatt (1958): a linear sum followed by a fixed scalar non-linearity. Real biological neurons possess complex dendritic trees, top-down predictive reflections, threshold-gated burst dynamics, and inhibitory polarity inversions. Meanwhile, mathematical theorems (Kolmogorov-Arnold, Tensor Manifolds) prove that edge-based continuous functions and hyper-dimensional tensors overcome fundamental limitations like spectral bias."*

Every architecture in this research collection:
1. **Has Closed-Form Analytical Gradients**: No approximations; fully verified via finite-difference checks ($< 10^{-3}$ relative error).
2. **Conforms to the Plug-and-Play API**: Compatible with [`Sequential`](../doraneural/model.py#L57), [`Adam`](../doraneural/optimizers.py#L175), schedulers, and loss functions.
3. **Supports Full Serialization**: Compatible with [`save_model`](../doraneural/serialization.py#L68) and [`load_model`](../doraneural/serialization.py#L106).
4. **Is Covered by Automated Unit Tests**: Complete test coverage in [`tests/test_neural_lib.py`](../tests/test_neural_lib.py).

---

## 📂 Executable Demos

All research experiments can be reproduced immediately using standalone example scripts:
- `python3 examples/explore_novel_neurons.py`: Single-layer XOR and non-linear function discovery.
- `python3 examples/benchmark_neuron_arena.py`: 3-task parameter-matched arena (Physics, Spirals, Spectral Bias).
- `python3 examples/explore_novel_transformers.py`: Standard vs Dendritic vs KAN Transformer sequence comparison.
- `python3 examples/explore_reflection_and_gating.py`: Bifurcated routing and cortical reflection demos.
- `python3 examples/explore_4d_and_inverter.py`: 4D Spacetime hyper-tensor networks and synaptic inverters.
- `python3 examples/train_bio_hybrid_model.py`: End-to-end Bio-Reflective KAN digit classifier.
- `python3 examples/explore_exotic_neurons.py`: Wave interference, chaotic dynamics, and quantum tunneling.
