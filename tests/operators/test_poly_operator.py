# operators/test_poly_operator.py

# import abc
import pytest
import numpy as np

# import scipy.linalg as la
# import scipy.sparse as sparse
# from scipy.special import comb

import opinf

from opinf.operators._polynomial_operator import PolynomialOperator

other_operators = opinf.operators._nonparametric


def test_instantiation():
    expected_polynomial_order = 0
    thingy = PolynomialOperator(polynomial_order=expected_polynomial_order)
    print(f"Successfully instantiated: {thingy}")
    assert thingy.polynomial_order == expected_polynomial_order


@pytest.mark.parametrize("r", np.random.randint(0, 100, size=(5,)))
def test_operator_dimension(r):

    # constant
    operator = PolynomialOperator(polynomial_order=0)
    assert (
        operator.my_operator_dimension(r=r)
        == other_operators.ConstantOperator.operator_dimension(r=r)
        == PolynomialOperator.operator_dimension(r=r, p=0)
    )

    # linear
    operator = PolynomialOperator(polynomial_order=1)
    assert (
        operator.my_operator_dimension(r=r)
        == PolynomialOperator.operator_dimension(r=r, p=1)
        == other_operators.LinearOperator.operator_dimension(r=r)
    )

    # quadratic
    operator = PolynomialOperator(polynomial_order=2)
    assert (
        operator.my_operator_dimension(r=r)
        == PolynomialOperator.operator_dimension(r=r, p=2)
        == other_operators.QuadraticOperator.operator_dimension(r=r)
    )

    # cubic
    operator = PolynomialOperator(polynomial_order=3)
    assert (
        operator.my_operator_dimension(r=r)
        == PolynomialOperator.operator_dimension(r=r, p=3)
        == other_operators.CubicOperator.operator_dimension(r=r)
    )


@pytest.mark.parametrize("r", [-1, -0.5, 3.5])
def test_operator_dimension_invalid_r(r):
    with pytest.raises(ValueError):
        PolynomialOperator.operator_dimension(r=r, p=0)


@pytest.mark.parametrize("p", [-1, -0.5, 3.5])
def test_operator_dimension_invalid_p(p):
    with pytest.raises(ValueError):
        PolynomialOperator.operator_dimension(r=2, p=p)


@pytest.mark.parametrize(
    "r,k", [(r, k) for r in range(1, 20) for k in [10, 20, 50, 100]]
)
def test_datablock_against_reference_implementation(r, k):
    state_ = np.random.random((r, k))

    # constant
    operator = PolynomialOperator(polynomial_order=0)
    datamatrix = operator.datablock(states=state_)
    operator_ref = other_operators.ConstantOperator()
    datamatrix_ref = operator_ref.datablock(states=state_)

    # check that the shape is correct
    assert (
        datamatrix.shape
        == datamatrix_ref.shape
        == (PolynomialOperator.operator_dimension(r=r, p=0), k)
    )

    # check that the entries are correct
    assert (datamatrix == datamatrix_ref).all()

    # linear
    operator = PolynomialOperator(polynomial_order=1)
    datamatrix = operator.datablock(states=state_)
    operator_ref = other_operators.LinearOperator()
    datamatrix_ref = operator_ref.datablock(states=state_)

    # check that the shape is correct
    assert (
        datamatrix.shape
        == datamatrix_ref.shape
        == (PolynomialOperator.operator_dimension(r=r, p=1), k)
    )

    # check that the entries are correct
    assert (datamatrix == datamatrix_ref).all()

    # quadratic
    operator = PolynomialOperator(polynomial_order=2)
    datamatrix = operator.datablock(states=state_)
    operator_ref = other_operators.QuadraticOperator()
    datamatrix_ref = operator_ref.datablock(states=state_)

    # check that the shape is correct
    assert (
        datamatrix.shape
        == datamatrix_ref.shape
        == (PolynomialOperator.operator_dimension(r=r, p=2), k)
    )

    # check that the entries are correct
    assert (datamatrix == datamatrix_ref).all()

    # cubic
    operator = PolynomialOperator(polynomial_order=3)
    datamatrix = operator.datablock(states=state_)
    operator_ref = other_operators.CubicOperator()
    datamatrix_ref = operator_ref.datablock(states=state_)

    # check that the shape is correct
    assert (
        datamatrix.shape
        == datamatrix_ref.shape
        == (PolynomialOperator.operator_dimension(r=r, p=3), k)
    )

    # check that the entries are correct
    assert (datamatrix == datamatrix_ref).all()
