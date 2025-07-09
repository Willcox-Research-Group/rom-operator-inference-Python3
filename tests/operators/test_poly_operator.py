# operators/test_poly_operator.py

import pytest
import numpy as np

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
    assert operator.operator_dimension(
        r=r
    ) == other_operators.ConstantOperator.operator_dimension(r=r)

    # linear
    operator = PolynomialOperator(polynomial_order=1)
    assert operator.operator_dimension(
        r=r
    ) == other_operators.LinearOperator.operator_dimension(r=r)

    # quadratic
    operator = PolynomialOperator(polynomial_order=2)
    assert operator.operator_dimension(
        r=r
    ) == other_operators.QuadraticOperator.operator_dimension(r=r)

    # cubic
    operator = PolynomialOperator(polynomial_order=3)
    assert operator.operator_dimension(
        r=r
    ) == other_operators.CubicOperator.operator_dimension(r=r)


@pytest.mark.parametrize("r", [-1, -0.5, 3.5])
def test_operator_dimension_invalid_r(r):
    op = PolynomialOperator(polynomial_order=1)
    with pytest.raises(ValueError):
        op.operator_dimension(r=r)


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
        == (operator.operator_dimension(r=r), k)
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
        == (operator.operator_dimension(r=r), k)
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
        == (operator.operator_dimension(r=r), k)
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
        == (operator.operator_dimension(r=r), k)
    )

    # check that the entries are correct
    assert (datamatrix == datamatrix_ref).all()


@pytest.mark.parametrize(
    "r,p", [(r, p) for r in range(1, 10) for p in range(4)]
)
def test_apply_against_reference(r, p):

    # test all operators with the same state
    _state = np.random.random((r,))

    # get random operator entries of the correct size
    if p == 0:
        _entries = np.random.random((r,))
    else:
        _entries = np.random.random(
            (r, PolynomialOperator(p).operator_dimension(r=r))
        )
        print(_entries.shape)

    # initialize operator and compute action on the state
    operator = PolynomialOperator(polynomial_order=p)
    operator.entries = _entries
    action = operator.apply(_state)

    # compare to code for the same polynomial order
    references = [
        other_operators.ConstantOperator(),
        other_operators.LinearOperator(),
        other_operators.QuadraticOperator(),
        other_operators.CubicOperator(),
    ]
    operator_ref = references[p]
    operator_ref.entries = _entries
    action_ref = operator_ref.apply(_state)

    # compare
    assert action.shape == action_ref.shape == (r,)  # same size
    assert np.isclose(action, action_ref).all()  # same entries
