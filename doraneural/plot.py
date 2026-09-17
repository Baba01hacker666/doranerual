"""Terminal ASCII training curve plotter.

Visualizes loss curves and accuracy trajectories directly in the console without
needing matplotlib or graphical display servers.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np


def plot_ascii_curve(
    values: List[float],
    title: str = "Training Progress",
    width: int = 48,
    height: int = 8,
    char: str = "█",
) -> str:
    """Render a numerical sequence as a clean terminal ASCII line graph.

    Args:
        values (List[float]): Sequence of metric values across epochs.
        title (str): Chart title.
        width (int): Character width of plot area.
        height (int): Character height of plot area.
        char (str): Plot point marker character.

    Returns:
        str: Formatted multi-line ASCII chart.
    """
    if not values:
        return "No data to plot."

    n_points = len(values)
    val_min = float(min(values))
    val_max = float(max(values))

    # Avoid zero division if values are flat
    if abs(val_max - val_min) < 1e-7:
        val_max += 0.1
        val_min -= 0.1

    # Resample values to fit width
    if n_points == 1:
        sampled_vals = [values[0]] * width
    else:
        indices = np.linspace(0, n_points - 1, width)
        sampled_vals = np.interp(indices, np.arange(n_points), values)

    # Initialize canvas matrix
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Plot points
    for col, v in enumerate(sampled_vals):
        # Normalized 0.0 to 1.0
        norm_v = (v - val_min) / (val_max - val_min)
        row = int(round(norm_v * (height - 1)))
        row = max(0, min(height - 1, row))
        # Invert row so high values are at top
        canvas[height - 1 - row][col] = char

    # Construct final display with axes and labels
    lines = []
    border = "─" * (width + 10)
    lines.append(f"┌{border}┐")
    lines.append(f"│ 📈 {title:<{width + 6}} │")
    lines.append(f"├{border}┤")

    for r in range(height):
        # Compute corresponding y-axis value
        y_val = val_max - (r / (height - 1)) * (val_max - val_min)
        y_label = f"{y_val:6.3f} ┤"
        row_content = "".join(canvas[r])
        lines.append(f"│ {y_label}{row_content} │")

    # Bottom axis
    axis_bot = " " * 8 + "└" + "─" * (width - 1)
    lines.append(f"│ {axis_bot} │")
    epoch_label = f"Epoch 1{' ' * (width - 12)}Epoch {n_points}"
    lines.append(f"│ {' ' * 8}{epoch_label:<{width}} │")
    lines.append(f"└{border}┘")

    return "\n".join(lines)


def plot_history(history: Any, width: int = 42, height: int = 7) -> str:
    """Plot both Loss and Accuracy side-by-side or stacked from a History object."""
    hist_dict = history.history if hasattr(history, "history") else history
    output = []

    if "loss" in hist_dict and hist_dict["loss"]:
        loss_chart = plot_ascii_curve(
            hist_dict["loss"],
            title=f"Loss Progress (Initial: {hist_dict['loss'][0]:.4f} ──▶ Final: {hist_dict['loss'][-1]:.4f})",
            width=width,
            height=height,
            char="█",
        )
        output.append(loss_chart)

    if "accuracy" in hist_dict and hist_dict["accuracy"]:
        acc_vals = hist_dict["accuracy"]
        acc_chart = plot_ascii_curve(
            [a * 100 for a in acc_vals],
            title=f"Accuracy Progress (Initial: {acc_vals[0]*100:.1f}% ──▶ Final: {acc_vals[-1]*100:.1f}%)",
            width=width,
            height=height,
            char="▲",
        )
        output.append(acc_chart)

    return "\n\n".join(output)
