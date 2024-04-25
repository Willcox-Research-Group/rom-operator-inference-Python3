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
