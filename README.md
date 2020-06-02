# Operator Inference

This is a Python implementation of Operator Inference for constructing projection-based reduced-order models of dynamical systems with a polynomial form.
The procedure is **data-driven** and **non-intrusive**, making it a viable candidate for model reduction of black-box or complex systems.
The methodology was introduced in [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
See [**References**](#references) for more papers that use or build on Operator Inference.

**Contributors**: [Renee Swischuk](https://github.com/swischuk), [Shane McQuarrie](https://github.com/shanemcq18), [Elizabeth Qian](https://github.com/elizqian), [Boris Kramer](https://github.com/bokramer), [Karen Willcox](https://kiwi.oden.utexas.edu/).

See [this repository](https://github.com/elizqian/operator-inference) for a MATLAB implementation and [DOCUMENTATION.md](DOCUMENTATION.md) for the code documentation.

## Problem Statement

Consider the (possibly nonlinear) system of _n_ ordinary differential equations with state variable **x**, input (control) variable **u**, and independent variable _t_:

<p align="center"><img src="https://raw.githubusercontent.com/swischuk/rom-operator-inference-Python3/master/img/prb/eq1.svg"/></p>

where

<p align="center"><img src="https://raw.githubusercontent.com/swischuk/rom-operator-inference-Python3/master/img/prb/eq2.svg"/></p>

This system is called the _full-order model_ (FOM).
If _n_ is large, as it often is in high-consequence engineering applications, it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is up to quadratic in the state **x** with optional linear control inputs **u**.
The procedure is data-driven, non-intrusive, and relatively inexpensive.
In the most general case, the code can construct and solve a reduced-order system with the polynomial form

<p align="center"><img src="https://raw.githubusercontent.com/swischuk/rom-operator-inference-Python3/master/img/prb/eq3.svg"/></p>

<!-- https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{\mathbf{c}}+\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{G}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)+\sum_{i=1}^m\hat{N}_{i}\hat{\mathbf{x}}(t)u_{i}(t),"/>
</p> -->

where now

<p align="center"><img src="https://raw.githubusercontent.com/swischuk/rom-operator-inference-Python3/master/img/prb/eq4.svg"/></p>
<p align="center"><img src="https://raw.githubusercontent.com/swischuk/rom-operator-inference-Python3/master/img/prb/eq5.svg"/></p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{G}\in\mathbb{R}^{r\times%20r^3},\qquad\hat{B}\in\mathbb{R}^{r\times%20m},\qquad\hat{N}_{i}\in\mathbb{R}^{r\times%20r}."/>
</p> -->

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low dimension _r_ << _n_.

See [DETAILS.md](DETAILS.md) for more mathematical details and an index of notation.


## Quick Start

#### Installation

Install from the command line with the following single command (requires [`pip`](https://pypi.org/project/pip/)).
```bash
$ pip3 install rom-operator-inference
```

#### Usage

Given a linear basis `Vr`, snapshot data `X`, and snapshot velocities `Xdot`, the following code learns a reduced model for a problem of the form _d**x**/dt = **c** + A**x**(t)_, then solves the reduced system for _0 ≤ t ≤ 1_.

```python
import numpy as np
import rom_operator_inference as roi

# Define a model of the form  dx / dt = c + Ax(t).
>>> model = roi.InferredContinuousROM(modelform="cA")

# Fit the model to snapshot data X, the snapshot derivative Xdot,
# and the linear basis Vr by solving for the operators c_ and A_.
>>> model.fit(Vr, X, Xdot)

# Simulate the learned model over the time domain [0,1] with 100 timesteps.
>>> t = np.linspace(0, 1, 100)
>>> X_ROM = model.predict(X[:,0], t)
```


## Examples

The [`examples/`](examples/) folder contains scripts and notebooks that set up and run several examples:
- [`examples/tutorial.ipynb`](https://nbviewer.jupyter.org/github/Willcox-Research-Group/rom-operator-inference-Python3/blob/master/examples/tutorial.ipynb): A walkthrough of a very simple heat equation example.
- [`examples/heat_1D.ipynb`](https://nbviewer.jupyter.org/github/Willcox-Research-Group/rom-operator-inference-Python3/blob/master/examples/heat_1D.ipynb): A more complicated one-dimensional heat equation example [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- [`examples/data_driven_heat.ipynb`](https://nbviewer.jupyter.org/github/Willcox-Research-Group/rom-operator-inference-Python3/blob/master/examples/data_driven_heat.ipynb): A purely data-driven example using data generated from a one-dimensional heat equation \[4\].
<!-- - `examples/TODO.ipynb`: Burgers' equation [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104). -->
<!-- - `examples/TODO.ipynb`: Euler equation [\[2\]](https://arc.aiaa.org/doi/10.2514/6.2019-3707). -->
<!-- This example uses MATLAB's Curve Fitting Toolbox to generate the random initial conditions. -->

(More examples coming)


## References

- \[1\] Peherstorfer, B. and Willcox, K.,
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
_Computer Methods in Applied Mechanics and Engineering_, Vol. 306, pp. 196-215, 2016.
([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{Peherstorfer16DataDriven,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
}</pre></details>

- \[2\] Qian, E., Kramer, B., Marques, A., and Willcox, K.,
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum & Exhibition, Dallas, TX, June 2019. ([Download](https://kiwi.oden.utexas.edu/papers/learn-data-driven-nonlinear-reduced-model-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    title     = {Transform \\& Learn: A data-driven approach to nonlinear model reduction},
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    booktitle = {AIAA Aviation 2019 Forum},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

- \[3\] Swischuk, R., Mainini, L., Peherstorfer, B., and Willcox, K.,
[Projection-based model reduction: Formulations for physics-based machine learning.](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
_Computers & Fluids_, Vol. 179, pp. 704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{swischuk2019projection,
    title     = {Projection-based model reduction: Formulations for physics-based machine learning},
    author    = {Swischuk, R. and Mainini, L. and Peherstorfer, B. and Willcox, K.},
    journal   = {Computers \\& Fluids},
    volume    = {179},
    pages     = {704--717},
    year      = {2019},
    publisher = {Elsevier}
}</pre></details>

- \[4\] Swischuk, R., [Physics-based machine learning and data-driven reduced-order modeling](https://dspace.mit.edu/handle/1721.1/122682). Master's thesis, Massachusetts Institute of Technology, 2019. ([Download](https://dspace.mit.edu/bitstream/handle/1721.1/122682/1123218324-MIT.pdf))<details><summary>BibTeX</summary><pre>
@phdthesis{swischuk2019physics,
    title  = {Physics-based machine learning and data-driven reduced-order modeling},
    author = {Swischuk, Renee},
    year   = {2019},
    school = {Massachusetts Institute of Technology}
}</pre></details>

- \[5\] Peherstorfer, B. [Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference](https://arxiv.org/abs/1908.11233). arXiv:1908.11233.
([Download](https://arxiv.org/pdf/1908.11233.pdf))<details><summary>BibTeX</summary><pre>
@article{peherstorfer2019sampling,
    title   = {Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference},
    author  = {Peherstorfer, Benjamin},
    journal = {arXiv preprint arXiv:1908.11233},
    year    = {2019}
}</pre></details>

- \[6\] Swischuk, R., Kramer, B., Huang, C., and Willcox, K., [Learning physics-based reduced-order models for a single-injector combustion process](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, published online March 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13. ([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2019_learning_ROMs_combustor,
    title   = {Learning physics-based reduced-order models for a single-injector combustion process},
    author  = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal = {AIAA Journal},
    volume  = {},
    pages   = {Published Online: 19 Mar 2020},
    url     = {},
    year    = {2020}
}</pre></details>

- \[7\] Qian, E., Kramer, B., Peherstorfer, B., and Willcox, K. [Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems](https://www.sciencedirect.com/science/article/abs/pii/S0167278919307651). _Physica D: Nonlinear Phenomena_, Volume 406, May 2020, 132401. ([Download](https://kiwi.oden.utexas.edu/papers/lift-learn-scientific-machine-learning-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{QKPW2020_lift_and_learn,
    title   = {Lift \\& Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems.},
    author  = {Qian, E. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {Physica {D}: {N}onlinear {P}henomena},
    volume  = {406},
    pages   = {132401},
    url     = {https://doi.org/10.1016/j.physd.2020.132401},
    year    = {2020}
}</pre></details>

- \[8\] Benner, P., Goyal, P., Kramer, B., Peherstorfer, B., and Willcox, K. [Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms](https://arxiv.org/abs/2002.09726). arXiv:2002.09726. Also Oden Institute Report 20-04. ([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-nonlinear-model-reduction-Benner-Goyal-Kramer-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{benner2020operator,
    title   = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
    author  = {Benner, P. and Goyal, P. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {arXiv preprint arXiv:2002.09726},
    year    = {2020}
}</pre></details>
