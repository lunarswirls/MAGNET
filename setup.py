from pathlib import Path
from setuptools import find_namespace_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="magnetpy",
    version="2026.04.03a",
    description="Multi-Axis Gradiometric Noise Elimination and Tracking (MAGNET) Python Package",
    author="Dany Waller",
    author_email="dany.c.waller@gmail.com",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["magnetpy", "magnetpy.*"]),
    python_requires=">=3.11, <4",
    install_requires=[
        "matplotlib>=3.10.7",
        "numpy>=2.4.0",
        "pandas>=2.3.3",
        "scipy>=1.16.3",
    ],
    extras_require={"dev": ["pytest>=8.3"]},
)
