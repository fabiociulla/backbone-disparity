# setup.py
from setuptools import setup, find_packages

setup(
    name="backbone-disparity",
    version="0.1.0",
    packages=find_packages(include=["backbone", "backbone.*"]),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "networkx>=3.0",
        "scikit-learn>=1.3",
        "python-louvain>=0.16",
        "infomap>=2.7",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7", "pytest-cov"],
    },
    python_requires=">=3.8",
)