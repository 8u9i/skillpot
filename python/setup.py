from setuptools import setup, find_packages

setup(
    name="axon-format",
    version="1.0.0",
    description="Adaptive eXecutable Object Notation — binary-first ML model container with zero-copy loading",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    author="Axon Project",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev": ["pytest", "pytest-benchmark"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: File Formats",
    ],
    url="https://github.com/8u9i/axon",
    project_urls={
        "Source": "https://github.com/8u9i/axon",
        "Documentation": "https://github.com/8u9i/axon/docs/spec.md",
    },
)
