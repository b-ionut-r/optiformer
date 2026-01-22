"""
OptiFormer-Lite: Transformer-based Hyperparameter Optimization
"""

from setuptools import setup, find_packages
from pathlib import Path


def parse_requirements(filename: str):
    """Parse requirements from a file, ignoring comments and empty lines."""
    requirements = []
    filepath = Path(__file__).parent / filename

    if not filepath.exists():
        return requirements

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Skip lines that are section headers (start with =)
            if line.startswith('='):
                continue
            # Handle inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:
                requirements.append(line)

    return requirements


# Parse requirements
all_requirements = parse_requirements('requirements.txt')

# Separate core requirements from real-world data packages
# Real-world data packages that might fail to install on some systems
REAL_WORLD_PACKAGES = {'yahpo-gym', 'openml', 'hpobench', 'ConfigSpace'}

core_requirements = []
realworld_requirements = []

for req in all_requirements:
    # Extract package name (before any version specifier)
    pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()

    if pkg_name in REAL_WORLD_PACKAGES:
        realworld_requirements.append(req)
    elif pkg_name not in {'black', 'isort', 'pytest', 'pytest-cov'}:
        core_requirements.append(req)


setup(
    name="optiformer",
    version="0.1.0",
    description="Transformer-based Hyperparameter Optimization",
    long_description=open('README.md').read() if Path('README.md').exists() else "",
    long_description_content_type="text/markdown",
    author="OptiFormer Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",

    # Core dependencies (always installed)
    install_requires=core_requirements,

    # Optional dependencies
    extras_require={
        # Development tools
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black",
            "isort",
        ],
        # Real-world HPO data sources (HIGHLY RECOMMENDED)
        # These are critical for model generalization
        "realworld": realworld_requirements,
        # Full installation (everything)
        "full": realworld_requirements + [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black",
            "isort",
        ],
    },

    # Entry points for CLI (if needed in future)
    entry_points={
        "console_scripts": [
            # "optiformer=optiformer.cli:main",
        ],
    },

    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="hyperparameter optimization, transformers, automl, machine learning",
)
