"""
Setup script for IDS Framework
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ids-framework",
    version="2.0.0",
    author="IDS Framework Team",
    author_email="contact@ids-framework.com",
    description="Advanced Deep Learning Based Intrusion Detection System with Blockchain Integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ids-framework/ids-framework",
    project_urls={
        "Bug Tracker": "https://github.com/ids-framework/ids-framework/issues",
        "Documentation": "https://ids-framework.readthedocs.io/",
        "Source Code": "https://github.com/ids-framework/ids-framework",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
        ],
        "monitoring": [
            "psutil>=5.8.0",
            "GPUtil>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ids-train=ids_framework.cli.train:main",
            "ids-evaluate=ids_framework.cli.evaluate:main",
            "ids-predict=ids_framework.cli.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ids_framework": [
            "configs/*.json",
            "data/*.txt",
        ],
    },
    keywords="intrusion detection, deep learning, lstm, blockchain, cybersecurity, nsl-kdd, machine learning",
    zip_safe=False,
)
