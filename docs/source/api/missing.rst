:orphan:

Generate Missing Stubs
======================

A bug in ``jupyter book``: ```autosummary``` rst blocks in ``.ipynb`` files do not generate the stub files for the summarized objects.
This file exists so that the missing pages are generated; it should **not** be included in ``_toc.yml``

lift.ipynb
----------

.. currentmodule:: opinf.lift

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   LifterTemplate
   QuadraticLifter
   PolynomialLifter

pre.ipynb
---------

.. currentmodule:: opinf.pre

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   shift
   scale
   ShiftScaleTransformer
   TransformerMulti
   TransformerTemplate

basis.ipynb
-----------

.. currentmodule:: opinf.basis

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   pod_basis
   cumulative_energy
   residual_energy
   svdval_decay
   LinearBasis
   PODBasis
   BasisMulti
   BasisTemplate

ddt.ipynb
---------

.. currentmodule:: opinf.ddt

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   DerivativeEstimatorTemplate
   UniformFiniteDifferencer
   NonuniformFiniteDifferencer
   InterpolationDerivativeEstimator
   fwd1
   fwd2
   fwd3
   fwd4
   fwd5
   fwd6
   bwd1
   bwd2
   bwd3
   bwd4
   bwd5
   bwd6
   ctr2
   ctr4
   ctr6
   ord2
   ord4
   ord6
   ddt_uniform
   ddt_nonuniform
   ddt

operators.ipynb
---------------

.. currentmodule:: opinf.operators

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   OperatorTemplate
   InputMixin
   OpInfOperator
   ConstantOperator
   LinearOperator
   QuadraticOperator
   CubicOperator
   InputOperator
   StateInputOperator
   ParametricOperatorTemplate
   ParametricOpInfOperator
   AffineConstantOperator
   AffineLinearOperator
   AffineQuadraticOperator
   AffineCubicOperator
   AffineInputOperator
   AffineStateInputOperator
   InterpConstantOperator
   InterpLinearOperator
   InterpQuadraticOperator
   InterpCubicOperator
   InterpInputOperator
   InterpStateInputOperator
   has_inputs
   is_nonparametric
   is_parametric
   is_uncalibrated

lstsq.ipynb
-----------

.. currentmodule:: opinf.lstsq

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   lstsq_size
   PlainSolver
   SolverTemplate
   L2Solver
   L2DecoupledSolver
   TikhonovSolver
   TikhonovDecoupledSolver
   TruncatedSVDSolver
   TotalLeastSquaresSolver
