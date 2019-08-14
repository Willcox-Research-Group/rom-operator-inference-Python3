# setup.py
import setuptools

with open("README.md", "r") as infile:
    readme_text = infile.read()

setuptools.setup(
    # Package name and version.
    name="rom_operator_inference-shanemcq18",
    version="0.3.1",

    # Package description, license, and keywords.
    description="Operator Inference for Data-Driven, Non-intrusive, Projection-based Model Reduction",
    license="MIT",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/shanemcq18/rom_operator_inference",
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
    author="Renee Swischuk et al.",
    author_email="swischuk@mit.edu",
    maintainer="Shane McQuarrie",
    maintainer_email="shanemcq@utexas.edu",

    # Technical details: source code, dependencies, test suite.
    packages=["rom_operator_inference"],
    install_requires=[
          "numpy",
          "scipy",
          "sklearn",
          "numba",
      ],
    setup_requires=["pytest-runner"],
    test_suite="pytest",
    tests_require=[
        "pytest",
        "numpy",
        "scipy",
    ],
)
