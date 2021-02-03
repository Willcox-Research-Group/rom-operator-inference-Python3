# _core/__init__.py
"""Classes and main functions."""

from ._base import *
from ._inferred import *
from ._intrusive import *
from ._interpolate import *
from ._affine import *


def select_model_class(time, rom_strategy, parametric=False):
    """Select the appropriate ROM model class for the situation.

    Parameters
    ----------
    time : str {"discrete", "continuous"}
        The type of full-order model to be reduced. Options:
        * "discrete": solve a discrete dynamical system,
          x_{j+1} = f(x_{j}, u_{j}), x_{0} = x0.
        * "continuous": solve an ordinary differential equation,
          dx / dt = f(t, x(t), u(t)), x(0) = x0.

    rom_strategy : str {"inferred", "intrusive"}
        Whether to use Operator Inference or intrusive projection to compute
        the operators of the reduced model. Options:
        * "inferred": use Operator Inference, i.e., solve a least-squares
          problem based on snapshot data.
        * "intrusive": use intrusive projection, i.e., project known full-order
          operators to the reduced space.

    parametric : str {"affine", "interpolated"} or False
        Whether or not the model depends on an external parameter, and how to
        handle the parametric dependence. Options:
        * False (default): the problem is nonparametric.
        * "affine": one or more operators in the problem depends affinely on
          the parameter, i.e., A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.
          Only valid for rom_strategy="intrusive".
        * "interpolated": construct individual models for each sample parameter
          and interpolate them for general parameter inputs. Only valid for
          rom_strategy="inferred", and only when the parameter is a scalar.

    Returns
    -------
    ModelClass : type
        One of the ROM classes derived from _BaseROM:
        * InferredDiscreteROM
        * InferredContinuousROM
        * IntrusiveDiscreteROM
        * IntrusiveContinuousROM
        * AffineIntrusiveDiscreteROM
        * AffineIntrusiveContinuousROM
        * InterpolatedInferredDiscreteROM
        * InterpolatedInferredContinuousROM
    """
    # Validate parameters.
    time_options = {"discrete", "continuous"}
    rom_strategy_options = {"inferred", "intrusive"}
    parametric_options = {False, "affine", "interpolated"}

    if time not in time_options:
        raise ValueError(f"input `time` must be one of {time_options}")
    if rom_strategy not in rom_strategy_options:
        raise ValueError(
                f"input `rom_strategy` must be one of {rom_strategy_options}")
    if parametric not in parametric_options:
        raise ValueError(
                f"input `parametric` must be one of {parametric_options}")

    t, r, p = time, rom_strategy, parametric

    if t == "discrete" and r == "inferred" and not p:
        return InferredDiscreteROM
    elif t == "continuous" and r == "inferred" and not p:
        return InferredContinuousROM
    elif t == "discrete" and r == "intrusive" and not p:
        return IntrusiveDiscreteROM
    elif t == "continuous" and r == "intrusive" and not p:
        return IntrusiveContinuousROM
    elif t == "discrete" and r == "intrusive" and p == "affine":
        return AffineIntrusiveDiscreteROM
    elif t == "continuous" and r == "intrusive" and p == "affine":
        return AffineIntrusiveContinuousROM
    elif t == "discrete" and r == "inferred" and p == "affine":
        return AffineInferredDiscreteROM
    elif t == "continuous" and r == "inferred" and p == "affine":
        return AffineInferredContinuousROM
    elif t == "discrete" and r == "inferred" and p == "interpolated":
        return InterpolatedInferredDiscreteROM
    elif t == "continuous" and r == "inferred" and p == "interpolated":
        return InterpolatedInferredContinuousROM
    else:
        raise NotImplementedError("model type invalid or not implemented")


def load_model(loadfile):
    """Load a serialized model from an HDF5 file, created previously from
    a ROM object's save_model() method.

    Parameters
    ----------
    loadfile : str
        The file to load from, which should end in '.h5'.

    Returns
    -------
    model : ROM class
        The trained reduced-order model.
    """
    import os, h5py

    if not os.path.isfile(loadfile):
        raise FileNotFoundError(loadfile)

    with h5py.File(loadfile, 'r') as data:
        if "meta" not in data:
            raise ValueError("invalid save format (meta/ not found)")
        if "operators" not in data:
            raise ValueError("invalid save format (operators/ not found)")

        # Load metadata.
        modelclass = data["meta"].attrs["modelclass"]
        try:
            ModelClass = eval(modelclass)
        except NameError as ex:
            raise ValueError(f"invalid modelclass '{modelclass}' (meta.attrs)")
        # is_parametric = issubclass(ModelClass, _ParametricMixin)
        modelform = data["meta"].attrs["modelform"]

        # Load basis if present.
        Vr = data["Vr"][:] if "Vr" in data else None

        # Load operators.
        operators = {}
        if 'c' in modelform:
            operators["c_"] = data["operators/c_"][:]
        if 'A' in modelform:
            operators["A_"] = data["operators/A_"][:]
        if 'H' in modelform:
            operators["H_"] = data["operators/H_"][:]
        if 'G' in modelform:
            operators["G_"] = data["operators/G_"][:]
        if 'B' in modelform:
            operators["B_"] = data["operators/B_"][:]

        # TODO: loading (and saving) for Parametric operators.

    # Construct the model.
    model = ModelClass(modelform).set_operators(Vr, **operators)

    return model
