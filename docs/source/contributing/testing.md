# Testing Source Code

This page is an overview of the package testing infrastructure.

:::{admonition} Summary
:class: note

- Use `tox -e format` to format the code with `black` conventions.
- Use `tox -e style` to verify that the code follows the `black` and `flake8` style guide.
- Write unit tests in the `tests/` folder that mirror the structure of the source code.
- Use `tox` to run unit tests.

The `Makefile` also has recipes for these commands, run `make` to see options.
:::

## Formatting with Black

Source and test code must conform to [`black`](https://black.readthedocs.io/en/stable/) conventions.

```shell
# Make sure the development environment is active.
$ conda deactivate
$ conda activate opinfdev

# Run the formatter.
(opinfdev) $ tox -e format
```

If the code was already formatted correctly, a report like the following will be printed.

```text
format: commands[0]> black src
All done! ✨ 🍰 ✨
46 files left unchanged.
format: commands[1]> black tests
All done! ✨ 🍰 ✨
41 files left unchanged.
  format: OK (0.33=setup[0.02]+cmd[0.18,0.14] seconds)
  congratulations :) (0.52 seconds)
```

If instead `black` made any changes ot the code, the report will show which files were changed.

```text
format: commands[0]> black src
reformatted /../OpInf/src/opinf/<the file that was changed>.py

All done! ✨ 🍰 ✨
1 file reformatted, 45 files left unchanged.
format: commands[1]> black tests
All done! ✨ 🍰 ✨
41 files left unchanged.
  format: OK (0.65=setup[0.03]+cmd[0.46,0.16] seconds)
  congratulations :) (0.89 seconds)
```

:::{note}
The `pre-commit` hook requires `black` and `flake8` to pass before allowing a `git commit`.

```shell
(opinfdev) $ python3 -m pre-commit install
```

:::

:::{tip}
Most IDEs, such as Visual Studio Code, have [plugins](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) for `black` so that code is automatically formatted when a file is saved.
:::

## Linting with Flake8

Source and test code must be free of syntax error and conform to the standard Python style guide, such as [PEP 8](https://www.python.org/dev/peps/pep-0008/).
Code style is verified by `black` and by [`flake8`](https://flake8.pycqa.org/en/latest/), a common Python [linter](https://en.wikipedia.org/wiki/Lint_(software)).
Unlike `black`, linters do not alter the code; they only checks that code satisfies the style guide and sometimes identifies syntax and other errors.

```shell
# Make sure the development environment is active.
$ conda deactivate
$ conda activate opinfdev

# Check code style.
$ tox -e style
```

If the code passes, a report like the following will be displayed.

```text
style: commands[0]> black --check src
All done! ✨ 🍰 ✨
46 files would be left unchanged.
style: commands[1]> flake8 src
style: commands[2]> black --check tests
All done! ✨ 🍰 ✨
41 files would be left unchanged.
style: commands[3]> flake8 tests
  style: OK (1.44=setup[0.05]+cmd[0.32,0.43,0.20,0.43] seconds)
  congratulations :) (1.67 seconds)
```

If the code fails, the report will provide specific feedback about what is wrong and where.
For example, suppose we add the following line somewhere in `src/opinf/pre/_base.py`.

```python
thisvariable = 'is never used'
```

This violates `black` conventions by using single quotation marks and `flake8` rules by defining a variable that is not used elsewhere.
Running `tox -e style` now gives a detailed report about the issue.

```text
style: commands[0]> black --check src
would reformat /../OpInf/src/opinf/pre/_base.py

Oh no! 💥 💔 💥
1 file would be reformatted, 45 files would be left unchanged.
style: exit 1 (0.42 seconds) /../OpInf> black --check src pid=XXXXX
style: commands[1]> flake8 src
src/opinf/pre/_base.py:38:9: F841 local variable 'thisvariable' is assigned to but never used
1     F841 local variable 'thisvariable' is assigned to but never used
style: exit 1 (0.50 seconds) /../OpInf> flake8 src pid=XXXXX
style: commands[2]> black --check tests
All done! ✨ 🍰 ✨
41 files would be left unchanged.
style: commands[3]> flake8 tests
  style: FAIL code 1 (1.58=setup[0.04]+cmd[0.42,0.50,0.17,0.46] seconds)
  evaluation failed :( (1.79 seconds)
```

:::{tip}
To see a more detailed report from `black` about the changes it would make, run `black --check --diff --color .`.

```text
--- /../OpInf/src/opinf/pre/_base.py
+++ /../OpInf/src/opinf/pre/_base.py
@@ -33,11 +33,11 @@
     """

     def __init__(self, name: str = None):
         """Initialize attributes."""

-        thisvariable = 'is never used'
+        thisvariable = "is never used"

         self.__n = None
         self.__name = name

     # Properties --------------------------------------------------------------
would reformat /../OpInf/src/opinf/pre/_base.py

Oh no! 💥 💔 💥
1 file would be reformatted, 45 files would be left unchanged.
```

:::

:::{warning}
Code that passes the style check is not guaranteed to _work_ as intended.
Linting does not replace unit tests for gauging code correctness and functionality.
:::

## Unit Testing with Pytest

This package uses the [Pytest](https://docs.pytest.org/en/7.0.x/) framework for unit testing.

- All tests are stored in the `tests/` folder.
- The file structure of `tests/` should mirror the file structure of `src/opinf/`, but test files must start with `test_`. For example, tests for the source file `src/opinf/pre/_basis.py` are grouped in `tests/pre/test_basis.py`. Within that file, the function `test_pod_basis()` runs tests for `pre.pod_basis()`.
- Tests for classes are grouped as classes. For example, the methods of the `TestBaseROM` class in `tests/roms/test_base.py` are unit tests for the methods of the `_BaseROM` class in `src/opinf/roms/_base.py`.

After making changes to the source code in `src/opinf` and writing corresponding tests in `tests/`, execute `tox` without any arguments in the command line from the root folder of the repository.

If all tests pass, a line coverage report will be generated.
Open `htmlcov/index.html` in a browser to view the report.

:::{note}
Running `tox` without any arguments tests the code for Python 3.9 through 3.12 (if they are installed on your system).
To test a single Python version, use `tox -e py310` for Python 3.10, `tox -e py311` for Python3.11, and so on.
:::

## GitHub Actions

Pull requests to `develop` are tested through GitHub Actions:
GitHub clones the new version of the repository, runs the linter and unit tests, compiles the documentation, and reports any errors.
**All tests must pass** before changes can be merged in.
Before pushing new changes, run `tox` until all tests pass.
