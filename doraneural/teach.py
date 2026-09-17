"""Educational guides, concept explanations, and learning tips for beginners.

Explains complex deep learning concepts in simple plain English without heavy math notation.
"""

from typing import Dict, List, Optional

GLOSSARY: Dict[str, Dict[str, str]] = {
    "neuron": {
        "title": "Neuron (Node)",
        "summary": "The fundamental building block that holds a single numerical value.",
        "details": (
            "A neuron receives numbers from previous neurons, multiplies each by a 'weight',\n"
            "adds a 'bias', and passes the result through an 'activation function' like ReLU."
        ),
        "analogy": "Like a recipe: inputs are ingredients, weights are how much of each ingredient to use.",
    },
    "weight": {
        "title": "Weights & Biases (Parameters)",
        "summary": "The adjustable dials the neural network turns to learn from data.",
        "details": (
            "Weights control the strength of connections between neurons.\n"
            "If an input is very important for predicting the answer, its weight becomes large.\n"
            "Biases let neurons fire even when inputs are zero."
        ),
        "analogy": "Like tuning the strings on a guitar until the sound is harmonious.",
    },
    "activation": {
        "title": "Activation Functions (ReLU, Sigmoid, Softmax)",
        "summary": "Non-linear switches that allow networks to learn complex, non-straight patterns.",
        "details": (
            "• ReLU: Turns negative numbers into 0. Simple, fast, and great for hidden layers.\n"
            "• Sigmoid: Squashes numbers into a 0 to 1 probability. Ideal for yes/no decisions.\n"
            "• Softmax: Turns a group of numbers into probabilities that sum to 100% (multiclass)."
        ),
        "analogy": "Like light switches that decide which signals are bright enough to pass through.",
    },
    "loss": {
        "title": "Loss Function (Error Metric)",
        "summary": "A score indicating how bad the network's current guesses are.",
        "details": (
            "When the network guesses wrong, the loss is high. As it learns, the loss drops close to 0.\n"
            "The entire goal of training is to make this loss number as small as possible."
        ),
        "analogy": "Like a golf score: the lower the number, the better your performance.",
    },
    "epoch": {
        "title": "Epochs",
        "summary": "One complete pass through the entire training dataset.",
        "details": (
            "If you have 1,000 training examples, 1 epoch means the network has seen and learned\n"
            "from all 1,000 examples once. Training for 20 epochs means it reviews the dataset 20 times."
        ),
        "analogy": "Like re-reading a textbook chapter several times before taking an exam.",
    },
    "learning_rate": {
        "title": "Learning Rate (lr)",
        "summary": "How big of a step the optimizer takes when adjusting weights.",
        "details": (
            "• Too high (e.g. 1.0): Network changes too wildly and may never learn (exploding loss).\n"
            "• Too low (e.g. 0.00001): Network learns so slowly it may take hours or get stuck.\n"
            "• Good default: 0.01 or 0.001."
        ),
        "analogy": "Walking down a mountain in the fog: giant leaps might make you fall, tiny baby steps take forever.",
    },
    "backprop": {
        "title": "Backpropagation (Chain Rule)",
        "summary": "The mechanism that traces errors backward to find out which weights caused the mistake.",
        "details": (
            "1. Forward Pass: Data moves left to right through the network to produce a guess.\n"
            "2. Error Calculation: Guess is compared to truth to see the mistake.\n"
            "3. Backward Pass: The error travels in reverse (right to left). Each weight gets a 'gradient'\n"
            "   showing whether increasing or decreasing it would reduce the error."
        ),
        "analogy": "Like a detective working backward from the scene of the crime to identify who contributed.",
    },
    "overfitting": {
        "title": "Overfitting vs Generalization",
        "summary": "When a network memorizes the training data instead of learning the underlying rule.",
        "details": (
            "Sign of overfitting: Training accuracy is 100%, but test accuracy drops significantly.\n"
            "How to fix: Use Dropout, reduce network size, or gather more diverse training data."
        ),
        "analogy": "Memorizing the exact answers to past test questions instead of learning math concepts.",
    },
}


def explain(topic: Optional[str] = None) -> str:
    """Return friendly, plain-English explanation for a machine learning concept."""
    if not topic or topic.lower() in ("list", "help", "all"):
        keys = ", ".join(GLOSSARY.keys())
        lines = [
            "🎓 doraneural Learning Center",
            "Available topics: " + keys,
            "\nUsage: doraneural explain <topic>   (e.g. doraneural explain backprop)",
        ]
        return "\n".join(lines)

    key = topic.lower().strip()
    # Fuzzy matching
    for k in GLOSSARY:
        if key in k or k in key:
            info = GLOSSARY[k]
            border = "─" * 62
            return (
                f"\n┌{border}┐\n"
                f"│ 📖 Topic: {info['title']:<50} │\n"
                f"├{border}┤\n"
                f"│ 📌 Quick Summary:                                          │\n"
                f"│    {info['summary']:<57} │\n"
                f"│                                                              │\n"
                f"│ 🔍 Details:                                                 │\n"
                f"│    " + info['details'].replace('\n', '\n│    ') + "\n"
                f"│                                                              │\n"
                f"│ 💡 Analogy:                                                 │\n"
                f"│    {info['analogy']:<57} │\n"
                f"└{border}┘"
            )

    return f"Topic '{topic}' not found. Available topics: {', '.join(GLOSSARY.keys())}"


def render_network_diagram(inputs: int, hidden: List[int], outputs: int) -> str:
    """Render a clean ASCII diagram of the network's layers."""
    all_layers = [f"Input\n({inputs})"] + [f"Hidden {i+1}\n({h})" for i, h in enumerate(hidden)] + [f"Output\n({outputs})"]
    
    boxes = []
    for layer_text in all_layers:
        lines = layer_text.split("\n")
        w = max(len(lines[0]), len(lines[1])) + 2
        top = f"┌{'─' * w}┐"
        mid1 = f"│ {lines[0]:<{w-2}} │"
        mid2 = f"│ {lines[1]:<{w-2}} │"
        bot = f"└{'─' * w}┘"
        boxes.append((top, mid1, mid2, bot))

    # Connect boxes horizontally with arrows
    arrow = " ──▶ "
    result = []
    for r in range(4):
        row_str = arrow.join([box[r] for box in boxes])
        result.append("  " + row_str)

    return "\n".join(result)
