# Testing Source Code

This page is an overview of the package testing infrastructure.

:::{admonition} Summary

- Use `make lint` to verify that code follows the style guide.
- Write unit tests in the `tests/` folder that mirror the structure of the source code.
- Use `make test` to run the tests.
:::

(sec-contrib-formatting)=
## Formatting with Black

:::{admonition} TODO

**This is a future feature**.
Some parts of the current package do not yet follow `black` conventions.

- What is `black`
- How to run it
- Pre-commit hook
- Format on save functionality in many IDEs
:::

(sec-contrib-linting)=
## Linting with Flake8

Before you can test your code, it must be free of syntax error and conform to the style guide.
We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and other standard Python formatting guidelines.
These are enforced by [`flake8`](https://flake8.pycqa.org/en/latest/), a common Python [linter](https://en.wikipedia.org/wiki/Lint_(software)).
The linter does not alter the code; it only checks that code satisfies a specified style guide and sometimes identifies syntax and other errors.

To use the linter, run `make lint` in the command line from the root folder of the repository.
If your code passes, you will see something like the following.

```bash
# Make sure your development environment is active.
$ conda deactivate
$ conda activate opinfdev

# Run the linter.
$ make lint
python3 -m tox -e lint
lint: commands[0]> flake8 src
lint: commands[1]> flake8 tests
  lint: OK (1.05=setup[0.03]+cmd[0.66,0.36] seconds)
  congratulations :) (1.96 seconds)
```

If your code fails, you will get specific feedback about what is wrong and where.
For example:

```bash
python3 -m flake8 src
src/opinf/roms/_base.py:29:5: E303 too many blank lines (3)
1     E303 too many blank lines (3)
make: *** [lint] Error 1
```

```{attention}
Code that passes the linter satisfies the style guide, but it is not guaranteed to _work_ as intended.
Linting does not replace unit tests for gauging code correctness and functionality.
```

## Unit Testing with Pytest

We use the [Pytest](https://docs.pytest.org/en/7.0.x/) framework for unit testing.

- All tests are stored in the `tests/` folder.
- The file structure of `tests/` should mirror the file structure of `src/opinf/`, but test files must start with `test_`. For example, tests for the source file `src/opinf/pre/_basis.py` are grouped in `tests/pre/test_basis.py`. Within that file, the function `test_pod_basis()` runs tests for `pre.pod_basis()`.
- Tests for classes are grouped as classes. For example, the methods of the `TestBaseROM` class in `tests/roms/test_base.py` are unit tests for the methods of the `_BaseROM` class in `src/opinf/roms/_base.py`.

After making changes to the source code in `src/opinf` and writing corresponding tests in `tests/`, execute `make test` in the command line from the root folder of the repository.

:::{margin}

```{note}
`make install` installs your local version of the package without running the linter or any tests.
`make test` always executes `make install` first to ensure that you are testing your development version of the source code.
```

:::

_Line coverage_ refers to the lines of code that tests have executed.

```{attention}
All code in `tests/` must also pass the linter.
Write readable tests for readable source code!
```

## GitHub Actions

Pull requests to `develop` are tested through GitHub Actions:
GitHub clones the new version of the repository, runs the tests, compiles the documentation, and reports any errors.
**All tests must pass** before your changes can be merged in.
Before pushing new changes, always `make test` and `make docs` until all tests pass and the documentation compiles without errors.
