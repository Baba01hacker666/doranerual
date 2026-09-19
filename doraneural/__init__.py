"""doraneural: A lightweight, modular neural network library & CLI built purely with NumPy.

Designed for educational hackability and low-power CPU environments.
"""

from .base import Layer
from .layers import (
    Dense,
    DendriticDense,
    ChebyshevKAN,
    BifurcatedDense,
    ReflectiveDense,
    InvertedDense,
    Tensor4DDense,
    ComplexWaveDense,
    FractalChaosDense,
    TunnelingDense,
    Dropout,
    LayerNorm,
    Flatten,
    Conv2D,
    MaxPool2D,
)
from .activations import ReLU, Sigmoid, Softmax, Tanh, SiLU, Inverter
from .losses import (
    Loss,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanSquaredError,
    MSELoss,
)
from .optimizers import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    clip_grad_norm,
    clip_grad_value,
)
from .schedulers import (
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    WarmupCosineLR,
)
from .recurrent import (
    SimpleRNN,
    RNN,
    LSTM,
    GRU,
)
from .attention import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    KANTransformerBlock,
    DendriticTransformerBlock,
)
from .metrics import Metric, Accuracy, accuracy_score, MSE, MAE, get_metric
from .model import Sequential, History, EvaluationResult
from .serialization import save_model, load_model
from .utils import (
    set_seed,
    train_test_split,
    one_hot_encode,
    to_categorical,
    batch_iterator,
    make_moons,
    make_blobs,
    make_digits,
)
from .accel import (
    set_im2col_backend,
    get_im2col_backend,
    is_numba_available,
    im2col_indices,
    col2im_indices,
)
from .precision import (
    set_precision,
    get_precision,
    precision_scope,
    to_precision,
    memory_summary,
)
from .dataloader import (
    DataLoader,
    Dataset,
    ArrayDataset,
)
from .easy import create, quick_train
from .teach import explain
from .data import load_csv, create_sample_classification_csv, create_sample_regression_csv
from .hf_dataset import download_hf_dataset, download_and_merge_hf_datasets, is_hf_dataset_identifier, get_hf_dataset_splits
from .image import (
    read_bmp,
    write_bmp,
    read_ppm,
    resize_image,
    load_image,
    load_image_dataset,
    render_image_ascii,
    create_sample_cat_dog_dataset,
)
from .autograd import (
    Tensor,
    Parameter,
    Module,
    Linear,
    tensor,
    no_grad,
    is_grad_enabled,
    mse_loss,
    binary_cross_entropy,
)
from .transformer import LoRAAdapter, TransformerDecoderLM, TransformerDecoderBlock, TensorAdamW
from .dora_norm import DoraNormBlock, DoraNormDecoderLM
from .jev import JevDecisionModel, JevChoiceHead, JevScoreHead, JevBooleanHead, rlcd_loss
from .pulse import ZexoPulse, ZexoXtra
from .compiler import (
    compile_model,
    CompiledModel,
    CompiledStep,
    StaticBufferPool,
)
from .dnb import (
    save_dnb,
    load_dnb,
    inspect_dnb,
)
from .llm import (
    LlamaLLM,
    LlamaConfig,
    LlamaTokenizer,
    HFTokenizer,
    load_pretrained_llm,
    download_hf_model,
    load_safetensors,
    load_hf_config,
)
from .chat import (
    ChatSession,
    ChatMessage,
)
from .cpp_backend import (
    CppLlamaEngine,
    is_cpp_available,
    build_cpp_library,
    get_cpu_arch,
    get_cpu_backend,
    print_cpu_features,
)
from .plot import plot_ascii_curve, plot_history
from .export import export_to_standalone_python
from .errors import DoraneuralError, ShapeMismatchError, ModelNotCompiledError

__version__ = "1.0.0"

__all__ = [
    # High-level beginner one-liners
    "create",
    "quick_train",
    "explain",
    # Data & Preprocessing
    "DataLoader",
    "Dataset",
    "ArrayDataset",
    "load_csv",
    "create_sample_classification_csv",
    "create_sample_regression_csv",
    "download_hf_dataset",
    "is_hf_dataset_identifier",
    "get_hf_dataset_splits",
    # Hardware Acceleration & JIT
    "set_im2col_backend",
    "get_im2col_backend",
    "is_numba_available",
    # Precision & Mixed-Precision Control
    "set_precision",
    "get_precision",
    "precision_scope",
    "to_precision",
    "memory_summary",
    # Image loading & Computer Vision (Zero-dependency)
    "read_bmp",
    "write_bmp",
    "read_ppm",
    "resize_image",
    "load_image",
    "load_image_dataset",
    "render_image_ascii",
    "create_sample_cat_dog_dataset",
    # Visual ASCII plotting
    "plot_ascii_curve",
    "plot_history",
    # Zero-dependency Exporter
    "export_to_standalone_python",
    # Base
    "Layer",
    # Layers
    "Dense",
    "DendriticDense",
    "ChebyshevKAN",
    "BifurcatedDense",
    "ReflectiveDense",
    "InvertedDense",
    "Tensor4DDense",
    "ComplexWaveDense",
    "FractalChaosDense",
    "TunnelingDense",
    "Dropout",
    "LayerNorm",
    "Flatten",
    "Conv2D",
    "MaxPool2D",
    # Recurrent Layers (Sequence Modeling)
    "SimpleRNN",
    "RNN",
    "LSTM",
    "GRU",
    # Attention & Transformers
    "PositionalEncoding",
    "MultiHeadAttention",
    "TransformerBlock",
    "KANTransformerBlock",
    "DendriticTransformerBlock",
    # Activations
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "SiLU",
    "Inverter",
    # Losses
    "Loss",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "MeanSquaredError",
    "MSELoss",
    # Optimizers & Gradient Clipping
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    "clip_grad_norm",
    "clip_grad_value",
    # Learning Rate Schedulers
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "WarmupCosineLR",
    # Metrics
    "Metric",
    "Accuracy",
    "accuracy_score",
    "MSE",
    "MAE",
    "get_metric",
    # Model & History
    "Sequential",
    "History",
    "EvaluationResult",
    # Autograd & Computational Graph
    "Tensor",
    "Parameter",
    "Module",
    "Linear",
    "tensor",
    "no_grad",
    "is_grad_enabled",
    "mse_loss",
    "binary_cross_entropy",
    # Full-transformer NumPy backpropagation
    "LoRAAdapter",
    "TransformerDecoderLM",
    "TransformerDecoderBlock",
    "TensorAdamW",
    "DoraNormBlock",
    "DoraNormDecoderLM",
    # Jev / Zexo-Pulse System-1 Non-Autoregressive Intelligence
    "JevDecisionModel",
    "JevChoiceHead",
    "JevScoreHead",
    "JevBooleanHead",
    "rlcd_loss",
    "ZexoPulse",
    "ZexoXtra",
    # Static Graph Compiler & AOT
    "compile_model",
    "CompiledModel",
    "CompiledStep",
    "StaticBufferPool",
    # Pretrained LLM & C++ Engine
    "LlamaLLM",
    "LlamaConfig",
    "LlamaTokenizer",
    "HFTokenizer",
    "load_pretrained_llm",
    "download_hf_model",
    "load_safetensors",
    "load_hf_config",
    "CppLlamaEngine",
    "is_cpp_available",
    "build_cpp_library",
    "get_cpu_arch",
    "get_cpu_backend",
    "print_cpu_features",
    # Interactive Chat & Context Manager
    "ChatSession",
    "ChatMessage",
    # Serialization (JSON/NPZ and Versioned Binary DNB)
    "save_model",
    "load_model",
    "save_dnb",
    "load_dnb",
    "inspect_dnb",
    # Utilities
    "set_seed",
    "train_test_split",
    "one_hot_encode",
    "to_categorical",
    "batch_iterator",
    "make_moons",
    "make_blobs",
    "make_digits",
    # Errors
    "DoraneuralError",
    "ShapeMismatchError",
    "ModelNotCompiledError",
]
