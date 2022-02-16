(sec-contrib-testing)=
# Testing Source Code

This page is an overview of the testing infrastructure for the package.

(sec-contrib-linting)=
## Linting with Flake8

Before you can test your code, it must be free of syntax error and conform to the style guide.
We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and other standard Python formatting guidelines.
These are enforced by [`flake8`](https://flake8.pycqa.org/en/latest/), a common Python [linter](https://en.wikipedia.org/wiki/Lint_(software)).
The linter does not alter the code: it checks that code satisfies a specified style guide and sometimes identifies syntax and other errors.

To use the linter, run `make lint` in the command line from the root folder of the repository.
If your code passes, you will see the following.
```bash
python3 -m flake8 src
python3 -m flake8 tests
```
If your code fails, you will get specific feedback about what is wrong and where.
For example:
```bash
python3 -m flake8 src
src/rom_operator_inference/core/_base.py:29:5: E303 too many blank lines (3)
1     E303 too many blank lines (3)
make: *** [lint] Error 1
```

```{important}
Code that passes the linter satisfies the style guide, but it is not guaranteed to _work_.
We still need to write tests to gauge code correctness and functionality.
```

## Unit Testing with Pytest

- Pytest reference
- `tests/` folder and its structure
- `make test`
- Coverage

## Regression Testing through Examples

Unit tests and coverage are important for keeping the code healthy, but it is even more important to test the code on real problems.
TODO

## GitHub Actions

When you make a pull request to `develop` or `main`, GitHub clones your version of the repository, runs the tests, compiles the documentation, and reports the results.
**All tests must pass** before your changes can be accepted.
Before pushing new changes, always `make test` and/or `make docs` until all tests pass and the documentation compiles without errors.
