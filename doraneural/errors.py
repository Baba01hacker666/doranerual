"""Friendly educational error messages and troubleshooting guides for beginners.

Transforms confusing matrix and tensor shape errors into clear, actionable advice.
"""

from typing import Optional, Tuple, Union


class DoraneuralError(Exception):
    """Base class for educational doraneural exceptions with plain-English fixes."""

    def __init__(self, title: str, explanation: str, how_to_fix: str) -> None:
        self.title = title
        self.explanation = explanation
        self.how_to_fix = how_to_fix
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        border = "─" * 60
        return (
            f"\n┌{border}┐\n"
            f"│ ❌ {self.title:<56} │\n"
            f"├{border}┤\n"
            f"│ 💡 What happened:                                         │\n"
            f"│    {self.explanation:<55} │\n"
            f"│                                                            │\n"
            f"│ 🛠️  How to fix it:                                         │\n"
            f"│    {self.how_to_fix:<55} │\n"
            f"└{border}┘"
        )


class ShapeMismatchError(DoraneuralError):
    """Raised when tensor or dataset shapes do not match what the layer expects."""

    def __init__(
        self,
        expected: Union[int, Tuple[int, ...]],
        got: Union[int, Tuple[int, ...]],
        layer_name: str = "Dense",
    ) -> None:
        title = f"Input Shape Mismatch in {layer_name}"
        explanation = (
            f"Layer expected inputs with {expected} features, "
            f"but received input data with shape {got}."
        )
        how_to_fix = (
            f"Adjust the layer's in_features to match your data, or ensure "
            f"your input has {expected} columns."
        )
        super().__init__(title, explanation, how_to_fix)


class ModelNotCompiledError(DoraneuralError):
    """Raised when calling fit or evaluate on an uncompiled model."""

    def __init__(self) -> None:
        title = "Model Not Compiled Yet"
        explanation = (
            "You asked the network to train, but haven't told it how to measure "
            "mistakes (loss) or learn (optimizer)."
        )
        how_to_fix = (
            "Call model.compile(loss=..., optimizer=...) or use dn.create() "
            "which compiles automatically."
        )
        super().__init__(title, explanation, how_to_fix)


class TargetShapeMismatchError(DoraneuralError):
    """Raised when ground truth target shapes don't match network output."""

    def __init__(self, target_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        title = "Target Labels Mismatch Network Output"
        explanation = (
            f"Your labels have shape {target_shape}, but network output is {output_shape}."
        )
        how_to_fix = (
            "For multiclass classification, use one_hot_encode(y, num_classes). "
            "For binary, use shape (N, 1)."
        )
        super().__init__(title, explanation, how_to_fix)
