# Installation

This page describes how to install `rom_operator_inference` locally.

````{tip}
To check if the package is installed, run the following in the command line.

```shell
$ python3 -m pip freeze | grep rom-operator-inference
rom-operator-inference==1.2.1
```

No output means the package was not found.
````

## Recommended: Latest Release

We recommend installing the package from [the Python Package Index](https://pypi.org/) with [`pip`](https://pypi.org/project/pip/).
This installs the [latest official release](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/releases).

```shell
$ python3 -m pip install --user rom-operator-inference
```

```{note}
Like most Python packages, `rom_operator_inference` has a few [software dependencies](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/network/dependencies).
To avoid conflicts with other installed packages, you may want to install [within a virtual environment](https://docs.python.org/3/tutorial/venv.html).
```

## Latest Commit to Main Branch

The following command installs the latest version from the master branch, which may or may not be associated with [an official release](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/releases).
This requires [`git`](https://git-scm.com/).

```shell
$ python3 -m pip install git+https://github.com/Willcox-Research-Group/rom-operator-inference-Python3.git
```

## Source Code

The final option is to download the entire repository (including tests, etc.) and install it from source.
This also requires [`git`](https://git-scm.com/) and is the first step for contributing.
<!-- TODO: link to contributing instructions and note on testing. -->

```shell
$ git clone https://github.com/Willcox-Research-Group/rom-operator-inference-Python3.git
$ python3 -m pip install rom-operator-inference-Python3
```
