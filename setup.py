from setuptools import setup, find_packages

setup(
    name="doraneural",
    version="1.0.0",
    description="A lightweight, modular neural network library & CLI tool built purely with NumPy for CPU and education.",
    author="Doranerual",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "doraneural=doraneural.cli:main",
            "doranerual=doraneural.cli:main",
        ],
    },
)
