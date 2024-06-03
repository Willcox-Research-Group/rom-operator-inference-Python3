# How to Contribute

Thank you for your interest in contributing to `opinf`!
Before you begin, please review our [Code of Conduct](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/blob/main/CODE_OF_CONDUCT.md).

:::{admonition} Summary

- Changes to the [source code](./code_anatomy.md) must be accompanied with updates to corresponding [unit tests](./testing.md) and [documentation](./documentation.md).
- Use `tox` to run tests while developing:
  - `tox -e style` checks that source code and tests follow the style guide.
  - `tox` (without arguments) executes all unit tests.
  - `tox -e literature,docs` compiles the documentation.
- When all tests pass, open a pull request to the `main` branch on GitHub.
:::

## Setup

:::{attention}
Contributing to this project requires familiarity with GitHub and `git`.
If you are unfamiliar with either, start with the [GitHub tutorial](https://docs.github.com/en/get-started/quickstart/hello-world) or the [git tutorial](https://git-scm.com/docs/gittutorial).
:::

Now that you are a `git` expert, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) [the GitHub repository](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) and [clone your fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository).
Add the original repository as [an upstream remote](https://docs.github.com/en/get-started/quickstart/fork-a-repo#configuring-git-to-sync-your-fork-with-the-original-repository).

```bash
git clone git@github.com:<username>/rom-operator-inference-Python3.git OpInf
cd OpInf
git remote add upstream git@github.com:Willcox-Research-Group/rom-operator-inference-Python3.git
```

Like most Python packages, `opinf` has a few [software dependencies](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/network/dependencies).
To avoid conflicts with other installed packages, we recommend installing `opinf` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (recommended) or [virtual Python environment](https://docs.python.org/3/tutorial/venv.html) .

```shell
# Make a fresh conda environment and install Python 3.12.
conda create -n opinfdev python=3.12
```

Be sure to [activate](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) the environment before using `pip` or other installation tools.
Then, install the package with developer dependencies.

```shell
# Activate the conda environment (updates the PATH).
$ conda activate opinfdev

# Verify python is now linked to the conda environment.
(opinfdev) $ which python3
/path/to/your/conda/envs/opinfdev/bin/python3
(opinfdev) $ python3 --version
Python 3.12.3

# Install the package and its dependencies in development mode.
(opinfdev) $ python3 -m pip install -e ".[dev]"
```

Style checks, unit tests, and documentation builds are managed with [`tox`](https://tox.wiki/en/latest).
For each of these tasks, `tox` creates a new virtual environment, installs the dependencies (e.g., `pytest` for running unit tests), and executes the task recipe.

:::{note}
Unit tests are executed for Python 3.9 through 3.12 if they are installed on your system.
The best way to install multiple Python versions varies by platform; for MacOS, [we suggest](https://stackoverflow.com/questions/36968425/how-can-i-install-multiple-versions-of-python-on-latest-os-x-and-use-them-in-par#answer-65094122) using [Homebrew](https://brew.sh/).

```shell
# After installing Homebrew:
brew install python@3.9
brew install python@3.10
brew install python@3.11
brew install python@3.12
```

:::

Finally, to ensure that new additions follow code standards and conventions, install the [git pre-commit hook](https://pre-commit.com/) with the following command.

```shell
(opinfdev) $ pre-commit install
```

:::{important}
Don't skip this step!
It will help prevent automated tests from failing when a pull request is made.
:::

## Branches and Workflow

The source repository has three special branches:

- `main` is the most up-to-date version of the code. Tags on the `main` branch correspond to [public PyPi releases](https://pypi.org/project/opinf/).
- `gh-pages` contains only the current build files for this documentation. _This branch is updated by maintainers only._
- `data` contains only data files used in documentation demos.

To contribute, get synced with the `main` branch, then start a new branch for making active changes.

```bash
git pull upstream main        # Synchronize main with the source repository.
git branch <mynewbranch>      # Create a new branch to make edits from.
git switch <mynewbranch>      # Switch to the new branch to do work.
```

You are now ready to make edits on your newly created local branch.
When you're ready, [create a pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request) to merge the changes into `Willcox-Research-Group:main`.

## Repository Organization

::::{margin}
:::{note}
Full examples, like the one listed on the left under the **Tutorials** heading, are part of the documentation.
They should be written as Jupyter notebooks and placed in `docs/content/tutorials/`.
:::
::::

The GitHub repository is organized as follows.

- [`src/opinf/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/src/opinf) contains the actual package code, see the [Source Code Guide](./code_anatomy.md).
- [`tests/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/tests) contains tests to be run with [`pytest`](https://docs.pytest.org/en/7.0.x/). The file structure of `tests/` should mirror the file structure of `src/opinf/`. See [Testing Source Code](./testing.md).
- [`docs/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/docs) contains documentation (including this page!). See [Documentation](./documentation.md).

## Acceptance Standards

Changes are not usually be accepted until the following tests pass.

1. `tox`: write or update tests to validate your additions or changes, preferably with full line coverage.
2. `tox -e style`: write readable code that conforms to our style guide.
3. `tox -e literature,docs`: write or update documentation based on your changes.

:::{tip}
The `Makefile` has recipes for these commands, run `make` to see options.
See [makefiletutorial.com](https://makefiletutorial.com/) for an overview of `Makefile` syntax and usage.
:::
