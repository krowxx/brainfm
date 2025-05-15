"""Setup script for BrainFM package."""

from setuptools import setup, find_packages

setup(
    name="brainfm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydub>=0.25.1",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "brainfm=brainfm.cli:main",
        ],
    },
    author="BrainFM Developers",
    author_email="example@example.com",
    description="Apply frequency modulation and effects to audio files",
    keywords="audio, modulation, brainfm",
    python_requires=">=3.6",
) 