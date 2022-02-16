# Getting Started

Thank you for your interest in contributing to `rom_operator_inference`!
Before you begin, please review our [Code of Conduct](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/blob/main/CODE_OF_CONDUCT.md).

## Setup

```{important}
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
- `develop` is the current development version of the code. **Contributors should make pull requests to `develop`, not `main`.**
- `gh-pages` is the same as `main` except that it also contains the current build files for this documentation. _This branch is updated by maintainers only._

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

- [`src/rom_operator_inference/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/src/rom_operator_inference) contains the actual package code. The code is divided into submodules:
    - `core`: operator and reduced-order model classes.
    - `lstsq`: solvers for the linear regression problem at the heart of Operator Inference.
    - `pre`: pre-processing tools (basis computation, state transformations, etc.).
    - `post`: post-processing tools (mostly error evaluation).
    - `utils`: other routines that are not important to casual users, but which advanced users may want access to.
- [`tests/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/tests) has tests to be run with [`pytest`](https://docs.pytest.org/en/7.0.x/). The file structure of `tests/` should mirror the file structure of `src/rom_operator_inference/`. See [Testing](sec-contrib-testing).
- [`docs/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/docs) has documentation (including this page!). See [Documentation](sec-contrib-docs).

```{note}
Full examples, like the one you see on the left under the **Examples** heading, are part of the documentation.
They should be written as Jupyter notebooks and placed in `docs/examples/`.
```

## Standards

For any changes to be accepted, you must
1. Write readable code that conforms to our standards. In particular, `make lint` must succeed.
2. Write or update tests to validate your additions or changes. In particular, `make test` must succeed with full line coverage.
3. Write or update documentation based on your changes. In particular, `make docs` must succeed.

More on each of these in [Testing](sec-contrib-testing) and [Documentation](sec-contrib-docs).
