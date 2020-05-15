# setup.py
import setuptools

with open("README.md", "r") as infile:
    readme_text = infile.read()

setuptools.setup(
    # Package name and version.
    name="rom_operator_inference",
    version="0.8.2",

    # Package description, license, and keywords.
    description="Operator inference for data-driven, non-intrusive model reduction of dynamical systems.",
    license="MIT",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/swischuk/rom-operator-inference-Python3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ],

    # Humans to contact about this code.
    author="Renee C. Swischuk et al.",
    maintainer="Shane A. McQuarrie",
    maintainer_email="shanemcq@utexas.edu",

    # Technical details: source code, dependencies, test suite.
    packages=["rom_operator_inference"],
    install_requires=[
        "python>=3.7",
        "h5py>=2.9.0",
        "numpy>=1.16",
        "scipy>=1.3",
        "matplotlib>=3.1",
        "scikit-learn>=0.18",
      ],
    setup_requires=["pytest-runner"],
    test_suite="pytest",
    tests_require=["pytest"],
)
