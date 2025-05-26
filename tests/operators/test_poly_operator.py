# operators/test_poly_operator.py

# import abc
import pytest
import numpy as np

# import scipy.linalg as la
# import scipy.sparse as sparse
# from scipy.special import comb

import opinf

from opinf.operators._poly_operator import PolyOperator

other_operators = opinf.operators._nonparametric


def test_instantiation():
    expected_polynomial_order = 0
    thingy = PolyOperator(polynomial_order=expected_polynomial_order)
    print(f"Successfully instantiated: {thingy}")
    assert thingy.polynomial_order == expected_polynomial_order


@pytest.mark.parametrize("r", np.random.randint(0, 100, size=(5,)))
def test_operator_dimension(r):

    # constant
    operator = PolyOperator(polynomial_order=0)
    assert (
        operator.my_operator_dimension(r=r)
        == other_operators.ConstantOperator.operator_dimension(r=r)
        == PolyOperator.operator_dimension(r=r, p=0)
    )

    # linear
    operator = PolyOperator(polynomial_order=1)
    assert (
        operator.my_operator_dimension(r=r)
        == PolyOperator.operator_dimension(r=r, p=1)
        == other_operators.LinearOperator.operator_dimension(r=r)
    )

    # quadratic
    operator = PolyOperator(polynomial_order=2)
    assert (
        operator.my_operator_dimension(r=r)
        == PolyOperator.operator_dimension(r=r, p=2)
        == other_operators.QuadraticOperator.operator_dimension(r=r)
    )

    # cubic
    operator = PolyOperator(polynomial_order=3)
    assert (
        operator.my_operator_dimension(r=r)
        == PolyOperator.operator_dimension(r=r, p=3)
        == other_operators.CubicOperator.operator_dimension(r=r)
    )


@pytest.mark.parametrize("r", [-1, -0.5, 3.5])
def test_operator_dimension_invalid_r(r):
    with pytest.raises(ValueError):
        PolyOperator.operator_dimension(r=r, p=0)


@pytest.mark.parametrize("p", [-1, -0.5, 3.5])
def test_operator_dimension_invalid_p(p):
    with pytest.raises(ValueError):
        PolyOperator.operator_dimension(r=2, p=p)
