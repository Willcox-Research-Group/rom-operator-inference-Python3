# core/interpolate/test_base.py
"""Tests for core.interpolate._base."""

import os
import h5py
import pytest
import numpy as np
import scipy.interpolate

import opinf


class BaseDummy(opinf.core.nonparametric._base._NonparametricOpInfROM):
    """Instantiable version of _NonparametricOpInfROM."""

    def predict(*args, **kwargs):
        return 100


class TestInterpolatedOpInfROM:
    """Test core.interpolate._base._InterpolatedOpInfROM."""

    class Dummy(opinf.core.interpolate._base._InterpolatedOpInfROM):
        """Instantiable version of _InterpolatedOpInfROM."""
        _LHS_ARGNAME = "ddts"
        _ModelClass = BaseDummy
        _ModelFitClass = BaseDummy

        def predict(*args, **kwargs):
            pass

    def test_init(self):
        """Test __init__()."""

        self.Dummy._ModelFitClass = float
        with pytest.raises(RuntimeError) as ex:
            self.Dummy("A")
        assert ex.value.args[0] == "invalid _ModelFitClass 'float'"
        self.Dummy._ModelFitClass = BaseDummy

        rom = self.Dummy("A", "cubicspline")
        assert rom.InterpolatorClass is scipy.interpolate.CubicSpline
        rom = self.Dummy("A", "linear")
        assert rom.InterpolatorClass is scipy.interpolate.LinearNDInterpolator
        rom = self.Dummy("A", float)
        assert rom.InterpolatorClass is float

        with pytest.raises(ValueError) as ex:
            self.Dummy("A", "other")
        assert ex.value.args[0] == "invalid InterpolatorClass 'other'"

    # Fitting -----------------------------------------------------------------
    def test_check_parameters(self):
        """Test _check_parameters()."""
        rom = self.Dummy("A")

        badparams = [[1, 2, 3], [4, 5]]
        with pytest.raises(ValueError) as ex:
            rom._check_parameters(badparams)
        assert ex.value.args[0] == \
            "parameter dimension inconsistent across samples"

        rom._check_parameters(list(range(10)))
        assert rom.s == len(rom) == 10
        assert rom.p == 1
        assert rom.InterpolatorClass is scipy.interpolate.CubicSpline

        rom._check_parameters([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert rom.s == len(rom) == 4
        assert rom.p == 2
        assert rom.InterpolatorClass is scipy.interpolate.LinearNDInterpolator

    def test_split_operator_dict(self, s=10):
        """Test _split_operator_dict()."""
        rom = self.Dummy("AB")
        rom._check_parameters(list(range(s)))

        # Null input.
        oplist = rom._split_operator_dict(None)
        assert isinstance(oplist, list)
        assert len(oplist) == s
        assert all(op is None for op in oplist)

        # Non-dictionary input.
        with pytest.raises(TypeError) as ex:
            rom._split_operator_dict(list("AB"))
        assert ex.value.args[0] == "known_operators must be a dictionary"

        # Bad dictionary values.
        with pytest.raises(TypeError) as ex:
            rom._split_operator_dict({"A": [1, 2, 3], "B": "B"})
        assert ex.value.args[0] == \
            ("known_operators must be a dictionary mapping a string to a "
             "list of ndarrays")

        # Inconsistent value lengths.
        with pytest.raises(ValueError) as ex:
            rom._split_operator_dict({"A": [1]*s, "B": [2]*(s+1)})
        assert ex.value.args[0] == \
            ("known_operators dictionary must map a modelform key to a list "
             f"of s = {s:d} ndarrays")

        # Proper usage.
        A = np.empty((3, 3))
        B = np.empty((3, 2))
        oplist = rom._split_operator_dict({"A": [None] + [A]*(s - 1), "B": B})
        assert len(oplist) == s
        for j, opdict in enumerate(oplist):
            assert isinstance(opdict, dict)
            if j == 0:
                assert len(opdict) == 1
            else:
                assert len(opdict) == 2
                assert opdict["A"] is A
            assert opdict["B"] is B

    def test_check_number_of_training_datasets(self, s=3):
        """Test _check_number_of_training_datasets()."""
        rom = self.Dummy("AB")
        rom._check_parameters(list(range(s)))

        with pytest.raises(ValueError) as ex:
            rom._check_number_of_training_datasets([(list(range(s + 1)),
                                                     "dummy")])
        assert ex.value.args[0] == \
            f"len(dummy) = {s+1:d} != {s:d} = len(parameters)"

        rom._check_number_of_training_datasets([(list(range(s)), "dummy2")])

    def test_process_fit_arguments(self, s=5):
        """Test _process_fit_arguments()."""
        rom = self.Dummy("AB")
        params = list(range(s))

        outs = rom._process_fit_arguments(None, params, None, None, None)
        for out in outs:
            assert isinstance(out, list)
            assert len(out) == s
            assert all(entry is None for entry in out)

        Vr = np.empty((10, 4))
        oplist = [{"A": 1, "B": 2}] * s
        outs = rom._process_fit_arguments(Vr, params, None, None, None,
                                          known_operators=oplist, solvers=1)
        assert outs[-2] is oplist
        regs = outs[-1]
        assert isinstance(regs, list)
        assert len(regs) == s
        assert all(reg == 1 for reg in regs)

    def test_interpolate_roms(self, m=3, s=4, r=6):
        """Test _interpolate_roms()."""
        params = list(range(s))

        # Wrong type.
        with pytest.raises(TypeError) as ex:
            self.Dummy("AB")._interpolate_roms(params, params)
        assert ex.value.args[0] == "expected roms of type BaseDummy"

        # Inconsistent ROM dimensions.
        B = np.random.random((r, m))
        roms = []
        for i in range(s):
            nonparametricrom = BaseDummy("AB")
            nonparametricrom.m = m
            if i == 0:
                nonparametricrom.r = r - 1
                nonparametricrom.A_ = np.random.random((r-1, r-1))
                nonparametricrom.B_ = B[1:, :]
            else:
                nonparametricrom.r = r
                nonparametricrom.A_ = np.random.random((r, r))
                nonparametricrom.B_ = B
            roms.append(nonparametricrom)

        with pytest.raises(ValueError) as ex:
            self.Dummy("cA")._interpolate_roms(params, roms)
        assert ex.value.args[0] == \
            "ROMs to interpolate must have modelform='cA'"

        with pytest.raises(ValueError) as ex:
            self.Dummy("AB")._interpolate_roms(params, roms)
        assert ex.value.args[0] == \
            "ROMs to interpolate must have equal dimensions (inconsistent r)"

        # Inconsistent input/control dimensions.
        nonparametricrom = BaseDummy("AB")
        nonparametricrom.r = r
        nonparametricrom.A_ = np.random.random((r, r))
        nonparametricrom.m = m - 1
        nonparametricrom.B_ = B[:, 1:]
        roms[0] = nonparametricrom

        with pytest.raises(ValueError) as ex:
            self.Dummy("AB")._interpolate_roms(params, roms)
        assert ex.value.args[0] == \
            "ROMs to interpolate must have equal dimensions (inconsistent m)"

        nonparametricrom = BaseDummy("AB")
        nonparametricrom.r = r
        nonparametricrom.A_ = np.random.random((r, r))
        nonparametricrom.m = m
        nonparametricrom.B_ = B
        roms[0] = nonparametricrom

        rom = self.Dummy("AB", "cubicspline")
        rom._interpolate_roms(params, roms)

        assert rom.r == r
        assert rom.m == m
        assert rom.c_ is None
        assert isinstance(rom.A_,
                          opinf.core.operators.InterpolatedLinearOperator)
        assert isinstance(rom.A_.interpolator, rom.InterpolatorClass)
        assert len(rom.A_.matrices) == s
        for i, nonparametricrom in enumerate(roms):
            assert np.all(rom.A_.matrices[i] == nonparametricrom.A_.entries)
        assert rom.H_ is None
        assert rom.G_ is None
        assert isinstance(rom.B_, opinf.core.operators.LinearOperator)
        assert np.all(rom.B_.entries == B)

    def test_fit(self, n=50, m=3, s=4, r=10, k=40):
        """Test fit()."""
        params = list(range(s))
        Vr = np.linalg.qr(np.random.standard_normal((n, k)))[0][:, :r]
        Qs = np.random.standard_normal((s, r, k))
        Qdots = np.random.standard_normal((s, r, k))
        Us = np.random.standard_normal((s, m, k))
        known = {"c": np.random.random(n)}
        c_ = Vr.T @ known["c"]
        reg = 1e-2

        # Get non-parametric OpInf ROMs for each parameter sample.
        nproms = [
            opinf.ContinuousOpInfROM("cAB").fit(Vr, Qs[i], Qdots[i], Us[i],
                                                known_operators=known,
                                                solver=reg)
            for i in range(s)
        ]

        # Do the parametric OpInf ROM fit with all parameters.
        rom = self.Dummy("cAB")
        rom.fit(Vr, params, Qs, Qdots, Us,
                known_operators=known, solvers=reg)

        # Check consistency with the non-parametric ROMs at training points.
        for i in range(s):
            assert isinstance(rom.c_, opinf.core.operators.ConstantOperator)
            assert np.all(rom.c_.entries == c_)
            assert isinstance(rom.A_,
                              opinf.core.operators.InterpolatedLinearOperator)
            assert np.all(rom.A_.matrices[i] == nproms[i].A_.entries)
            assert isinstance(rom.B_,
                              opinf.core.operators.InterpolatedLinearOperator)
            assert np.all(rom.B_.matrices[i] == nproms[i].B_.entries)

    def test_set_interpolator(self, m=2, s=5, r=8):
        """Test set_interpolator()."""
        params = list(range(s))
        rom = self.Dummy("AB")

        # Interpolate a set of ROMs.
        B = np.random.random((r, m))
        roms = []
        for i in range(s):
            nonparametricrom = BaseDummy("AB")
            nonparametricrom.r = r
            nonparametricrom.A_ = np.random.random((r, r))
            nonparametricrom.m = m
            nonparametricrom.B_ = B
            roms.append(nonparametricrom)
        rom = self.Dummy("AB", scipy.interpolate.CubicSpline)
        rom._interpolate_roms(params, roms)

        # Verify initial interpolation.
        assert isinstance(rom.A_,
                          opinf.core.operators.InterpolatedLinearOperator)
        assert isinstance(rom.A_.interpolator, scipy.interpolate.CubicSpline)
        assert len(rom.A_.matrices) == s
        for i, nonparametricrom in enumerate(roms):
            assert np.all(rom.A_.matrices[i] == nonparametricrom.A_.entries)
        assert rom.H_ is None
        assert rom.G_ is None
        assert isinstance(rom.B_, opinf.core.operators.LinearOperator)
        assert np.all(rom.B_.entries == B)

        # Change the interpolator and verify.
        rom.set_interpolator(scipy.interpolate.Akima1DInterpolator)
        assert isinstance(rom.A_.interpolator,
                          scipy.interpolate.Akima1DInterpolator)
        assert len(rom.A_.matrices) == s
        for i, nonparametricrom in enumerate(roms):
            assert np.all(rom.A_.matrices[i] == nonparametricrom.A_.entries)
        assert isinstance(rom.B_, opinf.core.operators.LinearOperator)
        assert np.all(rom.B_.entries == B)

    # Model persistence -------------------------------------------------------
    def test_save(self, s=5, n=15, m=2, r=3,
                  target="_interpsavemodeltest.h5"):
        """Test save()."""
        # Clean up after old tests.
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Get a test model.
        params = np.sort(np.random.random(s))
        Vr = np.random.random((n, r))
        ops = {
            "c": list(np.random.random((s, r))),
            "A": list(np.random.random((s, r, r))),
            "H": list(np.random.random((s, r, r*(r + 1)//2))),
            "G": list(np.random.random((s, r, r*(r + 1)*(r + 2)//6))),
            "B": list(np.random.random((s, r, m))),
        }

        def _trainedinterpmodel(modelform):
            """Get test model."""
            subops = {key: val for key, val in ops.items() if key in modelform}
            return self.Dummy(modelform).fit(Vr, params, None, None,
                                             known_operators=subops)

        def _checkfile(filename, rom, hasbasis):
            assert os.path.isfile(filename)
            with h5py.File(filename, 'r') as data:
                # Check metadata.
                assert "meta" in data
                assert len(data["meta"]) == 0
                assert data["meta"].attrs["modelform"] == rom.modelform

                # Check basis.
                if hasbasis:
                    assert "basis" in data
                    assert np.all(data["basis/entries"][:] == Vr)
                else:
                    assert "basis" not in data

                # Check parameters.
                assert "parameters" in data
                assert np.all(data["parameters"][:] == params)

                # Check operators
                assert "operators" in data
                if "c" in rom.modelform:
                    assert np.all(data["operators/c_"][:] == rom.c_.matrices)
                else:
                    assert "c_" not in data["operators"]
                if "A" in rom.modelform:
                    assert np.all(data["operators/A_"][:] == rom.A_.matrices)
                else:
                    assert "A_" not in data["operators"]
                if "H" in rom.modelform:
                    assert np.all(data["operators/H_"][:] == rom.H_.matrices)
                else:
                    assert "H_" not in data["operators"]
                if "G" in rom.modelform:
                    assert np.all(data["operators/G_"][:] == rom.G_.matrices)
                else:
                    assert "G_" not in data["operators"]
                if "B" in rom.modelform:
                    assert np.all(data["operators/B_"][:] == rom.B_.matrices)
                else:
                    assert "B_" not in data["operators"]

        rom = _trainedinterpmodel("cAB")
        rom.save(target, save_basis=False)
        _checkfile(target, rom, False)

        with pytest.raises(FileExistsError) as ex:
            rom.save(target, overwrite=False)
        assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

        rom.save(target, save_basis=True, overwrite=True)
        _checkfile(target, rom, True)

        rom = _trainedinterpmodel("c")
        rom.save(target, overwrite=True)
        _checkfile(target, rom, True)

        rom = _trainedinterpmodel("HG")
        rom.basis = None
        rom.save(target, save_basis=True, overwrite=True)
        _checkfile(target, rom, False)

        # Check that save() and load() are inverses.
        rom = _trainedinterpmodel("AB")
        rom.basis = Vr
        rom.save(target, save_basis=True, overwrite=True)
        rom2 = rom.load(target, rom.InterpolatorClass)
        assert rom2 is not rom
        assert rom2 == rom
        assert rom2.basis == rom.basis
        for attr in ["n", "m", "r", "modelform", "__class__"]:
            assert getattr(rom, attr) == getattr(rom2, attr)
        for attr in ["A_", "B_"]:
            got = getattr(rom2, attr)
            assert opinf.core.operators.is_operator(got)
            assert np.allclose(getattr(rom, attr).matrices, got.matrices)
        for attr in ["c_", "H_", "G_"]:
            assert getattr(rom, attr) is getattr(rom2, attr) is None

        # Check basis = None functionality.
        rom = _trainedinterpmodel("cH")
        rom.basis = None
        rom.save(target, overwrite=True)
        rom2 = rom.load(target, rom.InterpolatorClass)
        assert rom2 is not rom
        assert rom2 == rom
        for attr in ["m", "r", "modelform", "__class__"]:
            assert getattr(rom, attr) == getattr(rom2, attr)
        for attr in ["c_", "H_"]:
            got = getattr(rom2, attr)
            assert opinf.core.operators.is_operator(got)
            assert np.allclose(getattr(rom, attr).matrices, got.matrices)
        for attr in ["n", "A_", "B_", "G_", "basis"]:
            assert getattr(rom, attr) is getattr(rom2, attr) is None

        os.remove(target)

    def test_load(self, s=6, n=14, m=3, r=2,
                  target="_interploadmodeltest.h5"):
        """Test load()."""
        # Clean up after old tests if needed.
        if os.path.isfile(target):                  # pragma: no cover
            os.remove(target)

        # Get a test model.
        params = np.sort(np.random.random(s))
        Vr = np.random.random((n, r))
        ops = {
            "c": list(np.random.random((s, r))),
            "A": np.random.random((r, r)),
            "H": list(np.random.random((s, r, r*(r + 1)//2))),
            "G": list(np.random.random((s, r, r*(r + 1)*(r + 2)//6))),
            "B": list(np.random.random((s, r, m))),
        }
        InterpolatorClass = scipy.interpolate.CubicSpline

        # Make an empty HDF5 file to start with.
        with h5py.File(target, 'w'):
            pass

        with pytest.raises(ValueError) as ex:
            self.Dummy.load(target, None)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Make a partially compatible HDF5 file to start with.
        with h5py.File(target, 'a') as hf:
            # Store metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["modelform"] = "cAB"

        with pytest.raises(ValueError) as ex:
            self.Dummy.load(target, None)
        assert ex.value.args[0] == "invalid save format (operators/ not found)"

        # Store the arrays.
        with h5py.File(target, 'a') as hf:
            hf.create_dataset("operators/c_", data=ops["c"])
            hf.create_dataset("operators/A_", data=ops["A"])
            hf.create_dataset("operators/B_", data=ops["B"])

        with pytest.raises(ValueError) as ex:
            self.Dummy.load(target, None)
        assert ex.value.args[0] == \
            "invalid save format (parameters/ not found)"

        # Store the parameters.
        with h5py.File(target, 'a') as hf:
            hf.create_dataset("parameters", data=params)

        def _check_model(rom):
            assert isinstance(rom, self.Dummy)
            for attr in ["modelform",
                         "n", "r", "m",
                         "c_", "A_", "H_", "G_", "B_", "basis"]:
                assert hasattr(rom, attr)
            assert rom.modelform == "cAB"
            assert rom.r == r
            assert rom.m == m
            for attr in ["c_", "A_", "B_"]:
                assert opinf.core.operators.is_operator(getattr(rom, attr))
            assert np.allclose(rom.c_.matrices, ops['c'])
            assert np.all(rom.A_.entries == ops['A'])
            assert rom.H_ is None
            assert rom.G_ is None
            assert np.allclose(rom.B_.matrices, ops['B'])

        # Load the file correctly.
        rom = self.Dummy.load(target, InterpolatorClass)
        _check_model(rom)
        assert rom.basis is None

        # Add the basis and then load the file correctly.
        basis = opinf.basis.LinearBasis().fit(Vr)
        with h5py.File(target, 'a') as hf:
            hf["meta"].attrs["BasisClass"] = "LinearBasis"
            basis.save(hf.create_group("basis"))
        rom = self.Dummy.load(target, InterpolatorClass)
        _check_model(rom)
        assert isinstance(rom.basis, type(basis))
        assert np.all(rom.basis.entries == Vr)

        # One additional test to cover other cases.
        with h5py.File(target, 'a') as f:
            f["meta"].attrs["modelform"] = "HG"
            f.create_dataset("operators/H_", data=ops["H"])
            f.create_dataset("operators/G_", data=ops["G"])

        rom = self.Dummy.load(target, InterpolatorClass)
        assert isinstance(rom, self.Dummy)
        for attr in ["modelform",
                     "n", "r", "m",
                     "c_", "A_", "H_", "G_", "B_", "basis"]:
            assert hasattr(rom, attr)
        assert rom.modelform == "HG"
        assert rom.r == r
        assert rom.m == 0
        for attr in ["H_", "G_"]:
            assert opinf.core.operators.is_operator(getattr(rom, attr))
        assert rom.c_ is None
        assert rom.A_ is None
        assert np.allclose(rom.H_.matrices, ops["H"])
        assert np.allclose(rom.G_.matrices, ops["G"])
        assert rom.B_ is None
        assert isinstance(rom.basis, type(basis))
        assert np.all(rom.basis.entries == Vr)
        assert rom.n == n

        # Clean up.
        os.remove(target)
