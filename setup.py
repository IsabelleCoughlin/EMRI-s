from pathlib import Path
from setuptools import setup, find_packages

# Project root
root = Path(__file__).parent

# Long description from README if present
readme = root / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else "EMRI-s package"

setup(
    name="emri_s",                      # package name (adjust as needed)
    version="0.1.0",
    description="EMRI-s: tools for EMRI simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Isabelle Coughlin",
    url="https://github.com/IsabelleCoughlin/EMRI-s",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        # list runtime dependencies here, e.g.:
        # "numpy>=1.23",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        # Uncomment and edit if you want console scripts:
        # "console_scripts": ["emri-s=emri_s.cli:main"],
    },
)