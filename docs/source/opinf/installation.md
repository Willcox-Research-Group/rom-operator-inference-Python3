# Package Installation

This page describes how to install `opinf` locally.

:::{important}
Like most Python packages, `opinf` has a few [software dependencies](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/network/dependencies).
To avoid conflicts with other installed packages, we recommend installing `opinf` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (recommended) or [virtual Python environment](https://docs.python.org/3/tutorial/venv.html) .

```shell
# Make a fresh conda environment and install Python 3.11.
conda create -n opinf3.11 python=3.11
```

Be sure to [activate](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) the environment before using `pip` or other installation tools.

```shell
# Activate the conda environment (updates the PATH).
$ conda activate opinf3.11

# Verify python is now linked to the conda environment.
$ which python3
/path/to/your/conda/envs/opinf3.11/bin/python3
$ python3 --version
Python 3.11.8
```

:::

:::{tip}
To check if the package is already installed in the current Python environment, run the following in the command line.

```shell
$ python3 -m pip freeze | grep opinf
9:opinf==0.5.1
```

No output means the package was not found.
:::

## Latest Release from PyPi (Recommended)

We recommend installing the package from [the Python Package Index](https://pypi.org/) with [`pip`](https://pypi.org/project/pip/).
This installs the [latest official release](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/releases).

```shell
conda activate opinf3.11
python3 -m pip install opinf
```

## Latest Commit to Main Branch

The following command installs the latest version from the `main` branch, which may or may not be associated with [an official release](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/releases).
This requires [`git`](https://git-scm.com/).

```shell
conda activate opinf3.11
python3 -m pip install git+https://github.com/Willcox-Research-Group/rom-operator-inference-Python3.git
```

## Source Code

The final option is to download the entire repository (including tests, documentation, etc.) and install the package from source.
This also requires [`git`](https://git-scm.com/) and is the first step for contributing.
See the [Developer Guide](../contributing/how_to_contribute.md) if you are interested in contributing.

```shell
git clone https://github.com/Willcox-Research-Group/rom-operator-inference-Python3.git
conda activate opinf3.11
python3 -m pip install rom-operator-inference-Python3
```
