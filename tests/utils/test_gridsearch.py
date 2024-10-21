# utils/test_gridsearch.py
"""Test for utils._gridsearch."""

import abc
import pytest
import numpy as np

import opinf


gridsearch = opinf.utils.gridsearch


def test_gridsearch1D():
    """Test utils.gridsearch() for 1D problems."""

    def min1D(x):
        """One-dimensional function with minimizer x = 2."""
        xm3 = x - 3
        return 2.25 * xm3**4 + 0.5 * xm3**3 - 3.75 * xm3**2

    candidates = np.linspace(1.5, 4.5, 10)

    # Gridsearch only.
    xhat = gridsearch(min1D, candidates, gridsearch_only=True)
    assert isinstance(xhat, float)
    xloc = np.argmin(np.abs(candidates - xhat))
    assert xhat == candidates[xloc]

    # Gridsearch plus minimization.
    xhat = gridsearch(min1D, candidates, verbose=True)
    assert isinstance(xhat, float)
    assert np.isclose(xhat, 2)

    gridsearch(min1D, candidates.reshape((-1, 1)), verbose=False)

    # Limited candidates
    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        xhat = gridsearch(min1D, np.linspace(2, 4, 10), label="x")
    assert len(wn) == 2
    assert wn[0].message.args[0] == (
        "smallest x candidate won grid search, "
        "consider using smaller candidates"
    )
    assert wn[1].message.args[0] == (
        "x grid search performed better than optimization, "
        "falling back on grid search solution"
    )
    assert np.isclose(xhat, 2)

    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        xhat = gridsearch(
            min1D, np.linspace(0.5, 2, 10), label="y", verbose=True
        )
    assert len(wn) == 2
    assert wn[0].message.args[0] == (
        "largest y candidate won grid search, consider using larger candidates"
    )
    assert wn[1].message.args[0] == (
        "y grid search performed better than optimization, "
        "falling back on grid search solution"
    )
    assert np.isclose(xhat, 2)

    # Failed grid search
    with pytest.raises(RuntimeError) as ex:
        gridsearch(lambda x: np.inf, candidates, verbose=True, label="w")
    assert ex.value.args[0] == "w grid search failed"

    # Failed optimization
    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        gridsearch(min1D, [1e8], label="z")
    assert len(wn) == 1
    assert wn[0].message.args[0] == (
        "z grid search performed better than optimization, "
        "falling back on grid search solution"
    )


def test_gridsearchND(dim=4):
    """Test utils.gridsearch() for greater-than-one-dimensional problems."""
    np.random.seed(dim)

    def rosen(x):
        """The Rosenbrock function, minimizer at x = [1, 1, ..., 1]."""
        return np.sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        )

    candidates = np.concatenate(
        (10 ** np.random.uniform(-5, 2, size=(20, dim)), [[1.01] * dim])
    )

    # Gridsearch only.
    xhat = gridsearch(rosen, candidates, gridsearch_only=True)
    assert isinstance(xhat, np.ndarray)
    assert xhat.shape == (dim,)
    xloc = np.argmin(np.linalg.norm(candidates - xhat, axis=1))
    assert np.allclose(xhat, candidates[xloc])

    # Gridsearch plus minimization.
    xhat = gridsearch(rosen, candidates, verbose=True)
    assert isinstance(xhat, np.ndarray)
    assert xhat.shape == (dim,)
    assert np.allclose(xhat, 1, atol=3e-2)


class _TestRegTest(abc.ABC):
    """Tests for classes that inherit from _RegTest."""

    RegTestClass = NotImplemented
    uncopied_attrs = NotImplemented
    copied_attrs = NotImplemented

    @abc.abstractmethod
    def get_tests(self):
        """Yield test cases."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def get_stable_models(self):
        """Yield models that should be stable
        for the corresponding test case.
        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def get_unstable_models(self):
        """Yield models that should be **unstable**
        for the corresponding test case.
        """
        raise NotImplementedError  # pragma: no cover

    def test_evaluate(self):
        """Test evaluate()."""
        for test_case, stable, unstable in zip(
            self.get_tests(),
            self.get_stable_models(),
            self.get_unstable_models(),
        ):
            assert test_case.evaluate(stable)
            assert not test_case.evaluate(unstable)

    def test_copy(self):
        """Test copy()."""
        for test_case in self.get_tests():
            newICs = np.random.random(20)
            copy_case = test_case.copy(newICs)
            assert isinstance(copy_case, self.RegTestClass)

            def isequal(left, right):
                if left is None and right is None:
                    return True
                if not isinstance(right, type(left)):
                    return False
                if isinstance(left, (int, float, tuple)):
                    return left == right
                if isinstance(left, np.ndarray):
                    return left.shape == right.shape and np.allclose(
                        left, right
                    )
                if callable(left):
                    return callable(right) and right is left
                raise NotImplementedError(f"comparison for type {type(left)}")

            for attr in self.uncopied_attrs:
                assert hasattr(copy_case, attr)
                assert not isequal(
                    getattr(copy_case, attr), getattr(test_case, attr)
                )

            for attr in self.copied_attrs:
                assert hasattr(copy_case, attr)
                assert isequal(
                    getattr(copy_case, attr), getattr(test_case, attr)
                )


class TestDiscreteRegTest(_TestRegTest):
    RegTestClass = opinf.utils.DiscreteRegTest
    uncopied_attrs = ("initial_conditions",)
    copied_attrs = ("niters", "parameters", "inputs", "bound")

    def get_tests(self):
        """Yield test cases."""
        yield self.RegTestClass(
            np.ones(3),
            10,
            inputs=np.ones((2, 10)),
            bound=10,
        )
        yield self.RegTestClass(
            np.ones(3),
            100,
            parameters=np.ones(2),
            bound=200,
        )

    def get_stable_models(self):
        """Return a model that should be stable for the test case."""
        yield opinf.models.DiscreteModel(
            operators=[opinf.operators.InputOperator(np.zeros((3, 2)))]
        )
        yield opinf.models.ParametricDiscreteModel(
            operators=[
                opinf.operators.AffineLinearOperator(
                    coeffs=2,
                    entries=[np.zeros((3, 3)) for _ in range(2)],
                )
            ]
        )

    def get_unstable_models(self):
        """Return a model that should be unstable for the test case."""
        yield opinf.models.DiscreteModel(
            operators=[opinf.operators.InputOperator(1e6 * np.ones((3, 2)))]
        )
        yield opinf.models.ParametricDiscreteModel(
            operators=[
                opinf.operators.AffineLinearOperator(
                    coeffs=2,
                    entries=[5e6 * np.ones((3, 3)) for _ in range(2)],
                )
            ]
        )


class TestContinuousRegTest(_TestRegTest):
    RegTestClass = opinf.utils.ContinuousRegTest
    uncopied_attrs = ("initial_conditions",)
    copied_attrs = ("time_domain", "parameters", "input_function", "bound")

    def get_tests(self):
        """Yield test cases."""
        yield self.RegTestClass(
            np.ones(3),
            np.linspace(0, 10, 101),
            input_function=lambda t: np.ones(2),
            bound=10,
        )
        test_case = self.RegTestClass(
            np.ones(3),
            np.linspace(0, 10, 1001),
            parameters=np.ones(2),
            bound=200,
        )
        yield test_case
        yield test_case

    def get_stable_models(self):
        """Return a model that should be stable for the test case."""
        yield opinf.models.ContinuousModel(
            operators=[opinf.operators.InputOperator(np.zeros((3, 2)))]
        )
        model = opinf.models.ParametricContinuousModel(
            operators=[
                opinf.operators.AffineLinearOperator(
                    coeffs=2,
                    entries=[np.zeros((3, 3)) for _ in range(2)],
                )
            ]
        )
        yield model
        yield model

    def get_unstable_models(self):
        """Return a model that should be unstable for the test case."""
        yield opinf.models.ContinuousModel(
            operators=[opinf.operators.InputOperator(1e6 * np.ones((3, 2)))]
        )
        model = opinf.models.ParametricContinuousModel(
            operators=[
                opinf.operators.AffineLinearOperator(
                    coeffs=2,
                    entries=[5e6 * np.ones((3, 3)) for _ in range(2)],
                )
            ]
        )
        yield model
        model.predict = lambda *args, **kwargs: np.inf * np.ones((3, 1001))
        yield model


if __name__ == "__main__":
    pytest.main([__file__])
