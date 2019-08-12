import setuptools

with open("README.md", "r") as infile:
    readme_text = infile.read()

setuptools.setup(
    name="rom_operator_inference-shanemcq18",
    version="0.2.2",
    author="Renee Swischuk et al.",
    author_email="swischuk@mit.edu",
    maintainer="Shane McQuarrie",
    maintainer_email="shanemcq@utexas.edu",
    description="Operator Inference for Data-Driven, Non-intrusive, Projection-based Model Reduction",
    license="MIT",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/shanemcq18/rom_operator_inference",
    packages=["rom_operator_inference"],
    install_requires=[
          "numpy",
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
    ],
)
