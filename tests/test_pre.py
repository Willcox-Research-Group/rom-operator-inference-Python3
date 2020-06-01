# test_pre.py
"""Tests for rom_operator_inference.pre.py"""

import pytest
import numpy as np
from scipy import linalg as la
from collections import namedtuple
from matplotlib import pyplot as plt

import rom_operator_inference as roi


# Basis computation ===========================================================
@pytest.fixture
def set_up_basis_data():
    n = 2000
    k = 500
    return np.random.random((n,k)) - .5


def test_pod_basis(set_up_basis_data):
    """Test pre.pod_basis() on a small case with the ARPACK solver."""
    X = set_up_basis_data
    n,k = X.shape

    # Try with an invalid rank.
    rmax = min(n,k)
    with pytest.raises(ValueError) as exc:
        roi.pre.pod_basis(X, rmax+1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = {rmax+1} (need 1 <= r <= {rmax})"

    with pytest.raises(ValueError) as exc:
        roi.pre.pod_basis(X, -1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = -1 (need 1 <= r <= {rmax})"

    # Try with an invalid mode.
    with pytest.raises(NotImplementedError) as exc:
        roi.pre.pod_basis(X, None, mode="dense")
    assert exc.value.args[0] == "invalid mode 'dense'"

    U, vals, _ = la.svd(X, full_matrices=False)
    for r in [2, 10, rmax]:
        Ur = U[:,:r]
        vals_r = vals[:r]

        # Via scipy.linalg.svd().
        Vr, svdvals = roi.pre.pod_basis(X, r, mode="simple")
        assert Vr.shape == (n,r)
        assert np.allclose(Vr, Ur)
        assert svdvals.shape == (r,)
        assert np.allclose(svdvals, vals_r)

        # Via scipy.sparse.linalg.svds() (ARPACK).
        Vr, svdvals = roi.pre.pod_basis(X, r, mode="arpack")
        assert Vr.shape == (n,r)
        for j in range(r):      # Make sure the columns have the same sign.
            if not np.isclose(Ur[0,j], Vr[0,j]):
                Vr[:,j] = -Vr[:,j]
        assert np.allclose(Vr, Ur)
        assert svdvals.shape == (r,)
        assert np.allclose(svdvals, vals_r)

        # Via sklearn.utils.extmath.randomized_svd().
        Vr, svdvals = roi.pre.pod_basis(X, r, mode="randomized")
        assert Vr.shape == (n,r)
        # Light accuracy test (equality not guaranteed by randomized SVD).
        assert la.norm(np.abs(Vr) - np.abs(Ur)) < 5
        assert svdvals.shape == (r,)
        assert la.norm(svdvals - vals_r) < 3


def test_mean_shift(set_up_basis_data):
    """Test pre.mean_shift()."""
    X = set_up_basis_data
    xbar, Xshifted = roi.pre.mean_shift(X)
    assert xbar.shape == (X.shape[0],)
    assert Xshifted.shape == X.shape
    assert np.allclose(np.mean(Xshifted, axis=1), np.zeros(X.shape[0]))
    assert np.allclose(xbar.reshape((-1,1)) + Xshifted, X)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.mean_shift(np.random.random((3,3,3)))
    assert exc.value.args[0] == "data X must be two-dimensional"


# Reduced dimension selection =================================================
def test_significant_svdvals(set_up_basis_data):
    """Test pre.significant_svdvals()."""
    X = set_up_basis_data
    svdvals = la.svdvals(X)

    # Single cutoffs.
    r = roi.pre.significant_svdvals(svdvals, 1e-14, plot=False)
    assert isinstance(r, int) and r >= 1

    # Multiple cutoffss.
    rs = roi.pre.significant_svdvals(svdvals, [1e-10,1e-12], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, int) and r >= 1
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = roi.pre.significant_svdvals(svdvals, .0001, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = roi.pre.significant_svdvals(svdvals, [1e-4, 1e-8, 1e-12], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")


def test_energy_capture(set_up_basis_data):
    """Test pre.energy_capture()."""
    X = set_up_basis_data
    svdvals = la.svdvals(X)

    # Single threshold.
    r = roi.pre.energy_capture(svdvals, .9, plot=False)
    assert isinstance(r, np.int64) and r >= 1

    # Multiple thresholds.
    rs = roi.pre.energy_capture(svdvals, [.9, .99, .999], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, np.int64) and r >= 1
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = roi.pre.energy_capture(svdvals, .999, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = roi.pre.energy_capture(svdvals, [.9, .99, .999], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")


def test_projection_error(set_up_basis_data):
    """Test pre.projection_error()."""
    X = set_up_basis_data
    Vr = la.svd(X, full_matrices=False)[0][:,:X.shape[1]//3]

    err = roi.pre.projection_error(X, Vr)
    assert np.isscalar(err) and err >= 0


def test_minimal_projection_error(set_up_basis_data):
    """Test pre.minimal_projection_error()."""
    X = set_up_basis_data
    V = la.svd(X, full_matrices=False)[0][:,:X.shape[1]//3]

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.minimal_projection_error(np.ravel(X), V, 1e-14, plot=False)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad basis shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.minimal_projection_error(X, V[0], 1e-14, plot=False)
    assert exc.value.args[0] == "basis V must be two-dimensional"

    # Single cutoffs.
    r = roi.pre.minimal_projection_error(X, V, 1e-14, plot=False)
    assert isinstance(r, int) and r >= 1

    # Multiple cutoffs.
    rs = roi.pre.minimal_projection_error(X, V, [1e-10, 1e-12], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, int) and r >= 1
    assert rs == sorted(rs)

    # Plotting
    status = plt.isinteractive()
    plt.ion()
    roi.pre.minimal_projection_error(X, V, .0001, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    roi.pre.minimal_projection_error(X, V, [1e-4, 1e-6, 1e-10], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")


# Reprojection schemes ========================================================
def test_reproject_discrete(n=50, m=5, r=3):
    """Test pre.reproject_discrete()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
    I = np.eye(n)
    D = np.diag(1 - np.logspace(-1, -2, n))
    W = la.qr(np.random.normal(size=(n,n)))[0]
    A = W.T @ D @ W
    Ht = np.random.random((n,n,n))
    H = (Ht + Ht.T) / 20
    H = H.reshape((n, n**2))
    B = np.random.random((n,m))
    U = np.random.random((m,k))
    B1d = np.random.random(n)
    U1d = np.random.random(k)
    Vr = np.eye(n)[:,:r]
    x0 = np.zeros(n)
    x0[0] = 1

    # Try with bad initial condition shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.reproject_discrete(lambda x:x, Vr, x0[:-1], k)
    assert exc.value.args[0] == "basis Vr and initial condition x0 not aligned"

    # Linear case, no inputs.
    f = lambda x: A @ x
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("A").fit(Vr, X_)
    assert np.allclose(Vr @ X_, model.predict(X_[:,0], k))
    assert np.allclose(model.A_, Vr.T @ A @ Vr)

    # Linear case, 1D inputs.
    f = lambda x, u: A @ x + B1d * u
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k, U1d)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("AB").fit(Vr, X_, U1d)
    assert np.allclose(X_, Vr.T @ model.predict(X_[:,0], k, U1d))
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_.flatten(), Vr.T @ B1d)

    # Linear case, 2D inputs.
    f = lambda x, u: A @ x + B @ u
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k, U)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("AB").fit(Vr, X_, U)
    assert np.allclose(X_, Vr.T @ model.predict(X_[:,0], k, U))
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_, Vr.T @ B)

    # Quadratic case, no inputs.
    f = lambda x: A @ x + H @ np.kron(x,x)
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("AH").fit(Vr, X_)
    assert np.allclose(X_, Vr.T @ model.predict(X_[:,0], k))
    assert np.allclose(model.A_, Vr.T @ A @ Vr, atol=1e-6, rtol=1e-6)
    H_ = Vr.T @ H @ np.kron(Vr, Vr)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(model.H_ @ x2_, H_ @ x2_)


def test_reproject_continuous(n=100, m=20, r=10):
    """Test pre.reproject_continuous()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
    I = np.eye(n)
    D = np.diag(1 - np.logspace(-1, -2, n))
    W = la.qr(np.random.normal(size=(n,n)))[0]
    A = W.T @ D @ W
    Ht = np.random.random((n,n,n))
    H = (Ht + Ht.T) / 20
    H = H.reshape((n, n**2))
    B = np.random.random((n,m))
    U = np.random.random((m,k))
    B1d = np.random.random(n)
    U1d = np.random.random(k)
    Vr = np.eye(n)[:,:r]
    X = np.random.random((n,k))

    # Try with bad initial condition shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.reproject_continuous(lambda x:x, Vr, X[:-1,:])
    assert exc.value.args[0] == \
        f"X and Vr not aligned, first dimension {n-1} != {n}"

    # Linear case, no inputs.
    f = lambda x: A @ x
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("A").fit(Vr, X_, Xdot_)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)

    # Linear case, 1D inputs.
    f = lambda x, u: A @ x + B1d * u
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X, U1d)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("AB").fit(Vr, X_, Xdot_, U1d)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_.flatten(), Vr.T @ B1d)

    # Linear case, 2D inputs.
    f = lambda x, u: A @ x + B @ u
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X, U)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("AB").fit(Vr, X_, Xdot_, U)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_, Vr.T @ B)

    # Quadratic case, no inputs.
    f = lambda x: A @ x + H @ np.kron(x,x)
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("AH").fit(Vr, X_, Xdot_)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    H_ = Vr.T @ H @ np.kron(Vr, Vr)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(model.H_ @ x2_, H_ @ x2_)


# Derivative approximation ====================================================
DynamicState = namedtuple("DynamicState", ["time", "state", "derivative"])

def _difference_data(t):
    Y = np.row_stack((t,
                      t**2/2,
                      t**3/3,
                      np.sin(t),
                      np.exp(t),
                      1/(t+1),
                      t + t**2/2 + t**3/3 + np.sin(t) - np.exp(t)
                    ))
    dY = np.row_stack((np.ones_like(t),
                       t,
                       t**2,
                       np.cos(t),
                       np.exp(t),
                       -1/(t+1)**2,
                       1 + t + t**2 + np.cos(t) - np.exp(t)
                     ))
    return DynamicState(t, Y, dY)


@pytest.fixture
def set_up_uniform_difference_data():
    t = np.linspace(0, 1, 400)
    return _difference_data(t)


@pytest.fixture
def set_up_nonuniform_difference_data():
    t = np.linspace(0, 1, 400)**2
    return _difference_data(t)


def test_fwd4(set_up_uniform_difference_data):
    """Test pre._fwd4()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 5):
        # One-dimensional test.
        dY0 = roi.pre._fwd4(Y[0,j:j+5], dt)
        assert isinstance(dY0, float)
        assert np.isclose(dY0, dY[0,j])

        # Two-dimensional test.
        dYj = roi.pre._fwd4(Y[:,j:j+5].T, dt)
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])

        # Check agreement.
        assert dY0 == dYj[0]


def test_fwd6(set_up_uniform_difference_data):
    """Test pre._fwd6()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 7):
        # One-dimensional test.
        dY0 = roi.pre._fwd6(Y[0,j:j+7], dt)
        assert isinstance(dY0, float)
        assert np.isclose(dY0, dY[0,j])

        # Two-dimensional test.
        dYj = roi.pre._fwd6(Y[:,j:j+7].T, dt).T
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])

        # Check agreement.
        assert dY0 == dYj[0]


def test_xdot_uniform(set_up_uniform_difference_data):
    """Test pre.xdot_uniform()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for o in [2, 4, 6]:
        dY_ = roi.pre.xdot_uniform(Y, dt, order=o)
        assert dY_.shape == Y.shape
        assert np.allclose(dY, dY_, atol=1e-4)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.xdot_uniform(Y[:,0], dt, order=2)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad order.
    with pytest.raises(NotImplementedError) as exc:
        roi.pre.xdot_uniform(Y, dt, order=-1)
    assert exc.value.args[0] == "invalid order '-1'; valid options: {2, 4, 6}"

    # Try with bad dt type.
    with pytest.raises(TypeError) as exc:
        roi.pre.xdot_uniform(Y, np.array([dt, 2*dt]), order=-1)
    assert exc.value.args[0] == "time step dt must be a scalar (e.g., float)"


def test_xdot_nonuniform(set_up_nonuniform_difference_data):
    """Test pre.xdot_nonuniform()."""
    dynamicstate = set_up_nonuniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dY_ = roi.pre.xdot_nonuniform(Y, t)
    assert dY_.shape == Y.shape
    assert np.allclose(dY, dY_, atol=1e-4)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.xdot_nonuniform(Y[:,0], t)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad time shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.xdot_nonuniform(Y, np.dstack((t,t)))
    assert exc.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.pre.xdot_nonuniform(Y, np.hstack((t,t)))
    assert exc.value.args[0] == "data X not aligned with time t"


def test_xdot(set_up_uniform_difference_data,
              set_up_nonuniform_difference_data):
    """Test pre.xdot()."""
    # Uniform tests.
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]

    def _single_test(*args, **kwargs):
        dY_ = roi.pre.xdot(*args, **kwargs)
        assert dY_.shape == Y.shape
        assert np.allclose(dY, dY_, atol=1e-4)

    _single_test(Y, dt)
    _single_test(Y, dt=dt)
    for o in [2, 4, 6]:
        _single_test(Y, dt, o)
        _single_test(Y, dt, order=o)
        _single_test(Y, dt=dt, order=o)
        _single_test(Y, order=o, dt=dt)
        _single_test(Y, t)

    # Nonuniform tests.
    dynamicstate = set_up_nonuniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative

    _single_test(Y, t)
    _single_test(Y, t=t)

    # Try with bad arguments.
    with pytest.raises(TypeError) as exc:
        roi.pre.xdot(Y)
    assert exc.value.args[0] == \
        "at least one other argument required (dt or t)"

    with pytest.raises(TypeError) as exc:
        roi.pre.xdot(Y, order=2)
    assert exc.value.args[0] == \
        "keyword argument 'order' requires float argument dt"

    with pytest.raises(TypeError) as exc:
        roi.pre.xdot(Y, other=2)
    assert exc.value.args[0] == \
        "xdot() got unexpected keyword argument 'other'"

    with pytest.raises(TypeError) as exc:
        roi.pre.xdot(Y, 2)
    assert exc.value.args[0] == \
        "invalid argument type '<class 'int'>'"

    with pytest.raises(TypeError) as exc:
        roi.pre.xdot(Y, dt, 4, None)
    assert exc.value.args[0] == \
        "xdot() takes from 2 to 3 positional arguments but 4 were given"
