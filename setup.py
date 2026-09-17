from setuptools import setup, find_packages

setup(
    name="neural_lib",
    version="0.1.0",
    description="A lightweight, modular neural network library built purely with NumPy for CPU and education.",
    author="Doranerual",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)
