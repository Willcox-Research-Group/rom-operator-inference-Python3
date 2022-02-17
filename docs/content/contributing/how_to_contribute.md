(sec-contrib-start)=
# Getting Started

Thank you for your interest in contributing to `rom_operator_inference`!
Before you begin, please review our [Code of Conduct](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/blob/main/CODE_OF_CONDUCT.md).

:::{admonition} Summary
- Changes to the [source code](sec-contrib-anatomy) must be accompanied with updates to corresponding [unit tests](sec-contrib-testing) and [documentation](sec-contrib-docs).
- Use `Makefile` shortcuts while developing:
    - `make lint` checks that source code and tests follow the style guide.
    - `make test` executes all unit tests.
    - `make docs` compiles the documentation.
- When all tests pass, make a pull request to the `develop` branch.
:::

## Setup

```{attention}
Contributing to this project requires familiarity with GitHub and `git`.
If you are unfamiliar with either, start with the [GitHub tutorial](https://docs.github.com/en/get-started/quickstart/hello-world) or the [git tutorial](https://git-scm.com/docs/gittutorial).
```

Now that you are a git expert, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) [the GitHub repository](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) and [clone your fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository).
Add the original repository as [an upstream remote](https://docs.github.com/en/get-started/quickstart/fork-a-repo#configuring-git-to-sync-your-fork-with-the-original-repository).

```bash
$ git clone https://<username>@github.com/<username>/rom-operator-inference-Python3
$ cd rom-operator-inference-Python3
$ git remote add upstream https://<username>@github.com/Willcox-Research-Group/rom-operator-inference-Python3
```

## Branches and Workflow

The source repository has three special branches:
- `main` is reserved for [public PyPi releases](https://pypi.org/project/rom-operator-inference/). _This branch is updated by maintainers only._
- `develop` is the current development version of the code.
- `gh-pages` is the same as `main` except that it also contains the current build files for this documentation. _This branch is updated by maintainers only._

```{attention}
Contributors should make pull requests to the `develop` branch of `Willcox-Research-Group/rom-operator-inference-Python3`, not the `main` branch.
```

To contribute, get synced with the `develop` branch, then start a new branch for making active changes.

```bash
$ git branch develop            # Create a local develop branch.
$ git checkout develop          # Switch to the new local develop branch.
$ git pull upstream develop     # Update it with the upstream develop branch.
$ git branch <mynewbranch>      # Create a new branch to make edits from.
$ git checkout <mynewbranch>    # Switch to the working branch to do work.
```

You are now ready to make edits on your newly created local branch.
When you're ready, [create a pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request) to merge the changes into `Willcox-Research-Group:develop`.

## Repository Organization

The GitHub repository is organized as follows.

:::{margin}
```{note}
Full examples, like the one you see on the left under the **Examples** heading, are part of the documentation.
They should be written as Jupyter notebooks and placed in `docs/examples/`.
```
:::

- [`src/rom_operator_inference/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/src/rom_operator_inference) contains the actual package code, see the [Source Code Guide](sec-contrib-anatomy). The code is divided into submodules:
    - `core`: operator and reduced-order model classes.
    - `lstsq`: solvers for the linear regression problem at the heart of Operator Inference.
    - `pre`: pre-processing tools (basis computation, state transformations, etc.).
    - `post`: post-processing tools (mostly error evaluation).
    - `utils`: other routines that are not important to casual users, but which advanced users may want access to.
- [`tests/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/tests) contains tests to be run with [`pytest`](https://docs.pytest.org/en/7.0.x/). The file structure of `tests/` should mirror the file structure of `src/rom_operator_inference/`. See [Testing](sec-contrib-testing).
- [`docs/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/docs) contains documentation (including this page!). See [Documentation](sec-contrib-docs).

## Acceptance Standards

:::{margin}
```{tip}
The file `Makefile` defines routines that are triggered by the `make` command line utility.
The aptly named [makefiletutorial.com](https://makefiletutorial.com/) has a pretty good overview of `Makefile` syntax and usage.
```
:::

For any changes to be accepted, they need to address three things.
1. [**Source Code.**](sec-contrib-anatomy) Write readable code that conforms to our style guide. In particular, `make lint` must succeed.
2. [**Unit tests.**](sec-contrib-testing) Write or update tests to validate your additions or changes. In particular, `make test` must succeed with full line coverage.
3. [**Documentation.**](sec-contrib-docs) Write or update documentation based on your changes. In particular, `make docs` must succeed.
