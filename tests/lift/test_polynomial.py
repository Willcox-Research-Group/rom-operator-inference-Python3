# lift/test_polynomial.py
"""Tests for lift._polynomial.py."""


import pytest
import numpy as np

import opinf


class TestQuadraticLifter:
    """Test opinf.lift.QuadraticLifter."""

    Lifter = opinf.lift.QuadraticLifter

    def test_lift(self, n=10, k=20):
        """Test QuadraticLifter.lift()."""
        Q = np.random.random((n, k))
        Q_lifted = self.Lifter.lift(Q)
        assert Q_lifted.shape == (2 * n, k)
        assert np.all(Q_lifted[:n] == Q)

    def test_unlift(self, n=9, k=21):
        """Test QuadraticLifter.unlift()."""
        Q_lifted = np.random.random((2 * n, k))
        Q = self.Lifter.unlift(Q_lifted)
        assert Q.shape == (n, k)
        assert np.all(Q == Q_lifted[:n])

    def test_lift_ddts(self, n=11, k=19):
        """Test QuadraticLifter.lift_ddts()."""
        Q = np.random.random((n, k))
        dQ = np.random.random((n, k))
        dQ_lifted = self.Lifter.lift_ddts(Q, dQ)
        assert dQ_lifted.shape == (2 * n, k)

    def test_verify(self, n=6):
        """Run QuadraticLifter.verify()."""
        t = np.linspace(0, 1, 400)
        Q = np.array([np.sin(m * t) for m in range(1, n)])
        self.Lifter().verify(Q, t, tol=1e-4)


class TestPolynomialLifter:
    """Test opinf.lift.PolynomialLifter."""

    Lifter = opinf.lift.PolynomialLifter

    def test_init(self):
        """Test PolynomialLifter.__init__() and properties."""
        # Try with bad orders argument.
        with pytest.raises(TypeError) as ex:
            self.Lifter("applesauce")
        assert ex.value.args[0] == "'orders' must be a sequence of numbers"

        # One order.
        lifter = self.Lifter(10)
        assert lifter.orders == (10,)
        assert lifter.num_variables == 1

        # Multiple orders.
        lifter = self.Lifter((1, 2, 3))
        assert lifter.orders == (1, 2, 3)
        assert lifter.num_variables == 3

        # Non-invertible.
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Lifter(0)
        assert wn[0].message.args[0] == "q -> q^0 = 1 is not invertible"
        self.Lifter((0, 3))

    def test_str(self):
        """Test PolynomialLifter.__str__()."""
        lifter = self.Lifter(2)
        assert str(lifter) == "Lifting map q -> q^2"

        lifter = self.Lifter((-1, 3))
        assert str(lifter) == "Lifting map q -> (q^-1, q^3)"

    def test_lift(self, n=10, k=20):
        """Test PolynomialLifter.lift()."""
        Q = np.random.random((n, k))

        # q --> (q, q^2)
        Q_lifted = self.Lifter((1, 2)).lift(Q)
        assert Q_lifted.shape == (2 * n, k)
        assert np.all(Q_lifted[:n] == Q)
        assert np.all(Q_lifted[n:] == Q**2)

        # q --> (1, q^3, q^4)
        Q_lifted = self.Lifter((0, 3, 4)).lift(Q)
        assert Q_lifted.shape == (3 * n, k)
        assert np.all(Q_lifted[:n] == 1)
        assert np.all(Q_lifted[n : (2 * n)] == Q**3)
        assert np.all(Q_lifted[(2 * n) :] == Q**4)

        # q --> q
        Q_lifted = self.Lifter(1).lift(Q)
        assert Q_lifted.shape == Q.shape
        assert np.all(Q_lifted == Q)

        # q --> (1/q^2, 1/q, sqrt(q))
        Q_lifted = self.Lifter((-2, -1, 0.5)).lift(Q)
        assert Q_lifted.shape == (3 * n, k)
        assert np.allclose(Q_lifted[:n], 1 / Q**2)
        assert np.all(Q_lifted[n : (2 * n)] == 1 / Q)
        assert np.all(Q_lifted[(2 * n) :] == np.sqrt(Q))

    def test_unlift(self, n=9, k=21):
        """Test PolynomialLifter.unlift()."""
        # q --> (q, q^2)
        Q_lifted = np.random.random((2 * n, k))
        Q = self.Lifter((1, 2)).unlift(Q_lifted)
        assert Q.shape == (n, k)
        assert np.all(Q == Q_lifted[:n])

        # q --> q
        Q_lifted = np.random.random((n, k))
        Q = self.Lifter(1).unlift(Q_lifted)
        assert Q.shape == Q_lifted.shape
        assert np.all(Q == Q_lifted)

        # q --> (1/q, q^2)
        Q = np.random.random((n, k))
        Q_lifted = np.concatenate((1 / Q, Q**2))
        Q_unlifted = self.Lifter((-1, 2)).unlift(Q_lifted)
        assert Q_unlifted.shape == Q.shape
        assert np.allclose(Q_unlifted, Q)

        # Non-invertible.
        with pytest.warns(opinf.errors.OpInfWarning):
            lifter = self.Lifter(0)
        with pytest.raises(ZeroDivisionError) as ex:
            lifter.unlift(Q_lifted)
        assert ex.value.args[0] == "q -> q^0 = 1 is not invertible"

    def test_lift_ddts(self, n=11, k=19):
        """Test PolynomialLifter.lift_ddts()."""
        Q = np.random.random((n, k))
        dQ = np.random.random((n, k))

        # q --> (q, q^2)
        dQ_lifted = self.Lifter((1, 2)).lift_ddts(Q, dQ)
        assert dQ_lifted.shape == (2 * n, k)
        assert np.all(dQ_lifted[:n] == dQ)
        assert np.all(dQ_lifted[n:] == 2 * Q * dQ)

        # q --> (1, q^3, q^4)
        dQ_lifted = self.Lifter((0, 3, 4)).lift_ddts(Q, dQ)
        assert dQ_lifted.shape == (3 * n, k)
        assert np.all(dQ_lifted[:n] == 0)
        assert np.all(dQ_lifted[n : (2 * n)] == 3 * Q**2 * dQ)
        assert np.all(dQ_lifted[(2 * n) :] == 4 * Q**3 * dQ)

        # q --> q
        dQ_lifted = self.Lifter(1).lift_ddts(Q, dQ)
        assert dQ_lifted.shape == dQ.shape
        assert np.all(dQ_lifted == dQ)

        # q --> (1/q^2, 1/q, sqrt(q))
        dQ_lifted = self.Lifter((-2, -1, 0.5)).lift_ddts(Q, dQ)
        assert dQ_lifted.shape == (3 * n, k)
        assert np.allclose(dQ_lifted[:n], -2 / Q**3 * dQ)
        assert np.allclose(dQ_lifted[n : (2 * n)], -1 / Q**2 * dQ)
        assert np.allclose(dQ_lifted[(2 * n) :], 0.5 / np.sqrt(Q) * dQ)

    def test_verify(self, n=6):
        """Run PolynomialLifter.verify()."""
        t = np.linspace(0, 1, 2000)
        Q = np.abs(np.array([np.cos(m * t) for m in range(1, n)])) + 1
        for order in ((1, 2), (3, 5, 4, 2, 1), (-2, -1, 0.5)):
            print(order)
            self.Lifter(order).verify(Q, t)


if __name__ == "__main__":
    pytest.main([__file__])
