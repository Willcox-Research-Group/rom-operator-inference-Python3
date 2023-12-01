# How to Contribute

Thank you for your interest in contributing to `opinf`!
Before you begin, please review our [Code of Conduct](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/blob/main/CODE_OF_CONDUCT.md).

:::{admonition} Summary

- Changes to the [source code](./code_anatomy.md) must be accompanied with updates to corresponding [unit tests](./testing.md) and [documentation](./documentation.md).
- Use `Makefile` shortcuts while developing:
  - `make lint` checks that source code and tests follow the style guide.
  - `make test` executes all unit tests.
  - `make docs` compiles the documentation.
- When all tests pass, make a pull request to the `main` branch on GitHub.
:::

## Setup

:::{attention}
Contributing to this project requires familiarity with GitHub and `git`.
If you are unfamiliar with either, start with the [GitHub tutorial](https://docs.github.com/en/get-started/quickstart/hello-world) or the [git tutorial](https://git-scm.com/docs/gittutorial).
:::

Now that you are a git expert, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) [the GitHub repository](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) and [clone your fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository).
Add the original repository as [an upstream remote](https://docs.github.com/en/get-started/quickstart/fork-a-repo#configuring-git-to-sync-your-fork-with-the-original-repository).

```bash
git clone https://<username>@github.com/<username>/rom-operator-inference-Python3
cd rom-operator-inference-Python3
git remote add upstream https://<username>@github.com/Willcox-Research-Group/rom-operator-inference-Python3
```

## Branches and Workflow

The source repository has two special branches:

- `main` is the most up-to-date version of the code. Tags on the `main` branch correspond to [public PyPi releases](https://pypi.org/project/opinf/).
- `gh-pages` contains only the current build files for this documentation. _This branch is updated by maintainers only._

To contribute, get synced with the `main` branch, then start a new branch for making active changes.

```bash
git pull upstream main        # Synchronize main with the source repository.
git branch <mynewbranch>      # Create a new branch to make edits from.
git checkout <mynewbranch>    # Switch to the working branch to do work.
```

You are now ready to make edits on your newly created local branch.
When you're ready, [create a pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request) to merge the changes into `Willcox-Research-Group:main`.

## Repository Organization

The GitHub repository is organized as follows.

::::{margin}
:::{note}
Full examples, like the one you see on the left under the **Tutorials** heading, are part of the documentation.
They should be written as Jupyter notebooks and placed in `docs/content/tutorials/`.
:::
::::

- [`src/opinf/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/src/opinf) contains the actual package code, see the [Source Code Guide](./code_anatomy.md).
- [`tests/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/tests) contains tests to be run with [`pytest`](https://docs.pytest.org/en/7.0.x/). The file structure of `tests/` should mirror the file structure of `src/opinf/`. See [Testing](./testing.md).
- [`docs/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/docs) contains documentation (including this page!). See [Documentation](./documentation.md).

## Acceptance Standards

::::{margin}
:::{tip}
The file `Makefile` defines routines that are triggered by the `make` command line utility.
The aptly named [makefiletutorial.com](https://makefiletutorial.com/) gives a good overview of `Makefile` syntax and usage.
:::
::::

For any changes to be accepted, they need to address three things.

1. [**Source Code.**](./code_anatomy.md) Write readable code that conforms to our style guide: `make lint` must succeed.
2. [**Unit tests.**](./testing.md) Write or update tests to validate your additions or changes: `make test` must succeed with full line coverage.
3. [**Documentation.**](./documentation.md) Write or update documentation based on your changes: `make docs` must succeed.
