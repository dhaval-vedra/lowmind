"""
LowMind — Ultra-Lightweight Deep Learning Framework
"""
from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lowmind",
    version="2.0.0",
    description="Ultra-lightweight deep learning framework for Raspberry Pi and low-end devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dhaval Vedra",
    author_email="",
    url="https://github.com/dhaval-vedra/lowmind",
    project_urls={
        "Bug Tracker": "https://github.com/dhaval-vedra/lowmind/issues",
        "Source Code": "https://github.com/dhaval-vedra/lowmind",
        "Documentation": "https://github.com/dhaval-vedra/lowmind#readme",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
        "plot": [
            "matplotlib>=3.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "deep learning", "neural network", "machine learning",
        "raspberry pi", "embedded", "edge ai", "autograd",
        "numpy", "lightweight", "iot",
    ],
    license="MIT",
    include_package_data=True,
)
