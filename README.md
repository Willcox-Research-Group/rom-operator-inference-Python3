[![License](https://img.shields.io/github/license/Willcox-Research-Group/rom-operator-inference-python3)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/Willcox-Research-Group/rom-operator-inference-python3)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/Willcox-Research-Group/rom-operator-inference-python3)
[![Issues](https://img.shields.io/github/issues/Willcox-Research-Group/rom-operator-inference-python3)](https://github.com/Willcox-Research-Group/rom-operator-inference-python3/issues)
[![Latest commit](https://img.shields.io/github/last-commit/Willcox-Research-Group/rom-operator-inference-python3)](https://github.com/Willcox-Research-Group/rom-operator-inference-python3/commits/main)
[![Documentation](https://img.shields.io/badge/Documentation-WIKI-important)](https://github.com/Willcox-Research-Group/rom-operator-inference-python3/wiki)

# Operator Inference in Python

This is a Python implementation of Operator Inference for learning projection-based polynomial reduced-order models of dynamical systems.
The procedure is **data-driven** and **non-intrusive**, making it a viable candidate for model reduction of "glass-box" systems.
The methodology was introduced in [\[1\]](#references).

[**See the Wiki**](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/wiki) for mathematical details and API documentation.
See [this repository](https://github.com/Willcox-Research-Group/rom-operator-inference-MATLAB) for a MATLAB implementation.

## Quick Start

### Installation

Install the package from the command line with the following single command (requires [`pip`](https://pypi.org/project/pip/)).
```bash
$ python3 -m pip install --user rom-operator-inference
```
[**See the wiki**](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/wiki) for other installation options.

### Usage

Given a basis matrix `Vr`, snapshot data `Q`, and snapshot time derivatives `Qdot`, the following code learns a reduced-order model for a problem of the form d**q** / dt = **c** + **Aq**(t), then solves the reduced system for 0 ≤ t ≤ 1.

```python
import numpy as np
import rom_operator_inference as opinf

# Define a reduced-order model of the form  dq / dt = c + Aq(t).
>>> rom = opinf.InferredContinuousROM(modelform="cA")

# Fit the model to snapshot data Q, the time derivatives Qdot,
# and the linear basis Vr by solving for the operators c_ and A_.
>>> rom.fit(Vr, Q, Qdot)

# Simulate the learned model over the time domain [0,1] with 100 timesteps.
>>> t = np.linspace(0, 1, 100)
>>> Q_ROM = rom.predict(Q[:,0], t)
```

---

**Contributors**:
[Shane McQuarrie](https://github.com/shanemcq18),
[Renee Swischuk](https://github.com/swischuk),
[Elizabeth Qian](https://github.com/elizqian),
[Boris Kramer](http://kramer.ucsd.edu/),
[Karen Willcox](https://kiwi.oden.utexas.edu/).


## References

These publications introduce, build on, or use Operator Inference.
Entries are listed chronologically.

- \[1\] [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ) and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Data-driven operator inference for non-intrusive projection-based model reduction.**](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
_Computer Methods in Applied Mechanics and Engineering_, Vol. 306, pp. 196-215, 2016.
([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{PW2016OperatorInference,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
}</pre></details>

- \[2\] [Qian, E.](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Marques, A.](https://scholar.google.com/citations?user=d4tBWWwAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Transform & Learn: A data-driven approach to nonlinear model reduction**](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum & Exhibition, Dallas, TX, June 2019. Paper AIAA-2019-3707.
([Download](https://kiwi.oden.utexas.edu/papers/learn-data-driven-nonlinear-reduced-model-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019TransformAndLearn,
    title     = {Transform \\& Learn: A data-driven approach to nonlinear model reduction},
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    booktitle = {AIAA Aviation 2019 Forum},
    year      = {2018},
    address   = {Dallas, TX},
    note      = {Paper AIAA-2019-3707},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

- \[3\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Mainini, L.](https://scholar.google.com/citations?user=1mo8GgkAAAAJ), [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Projection-based model reduction: Formulations for physics-based machine learning.**](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
_Computers & Fluids_, Vol. 179, pp. 704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SMPW2019PhysicsbasedML,
    title     = {Projection-based model reduction: Formulations for physics-based machine learning},
    author    = {Swischuk, R. and Mainini, L. and Peherstorfer, B. and Willcox, K.},
    journal   = {Computers \\& Fluids},
    volume    = {179},
    pages     = {704--717},
    year      = {2019},
    publisher = {Elsevier}
}</pre></details>

- \[4\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ),
[**Physics-based machine learning and data-driven reduced-order modeling**](https://dspace.mit.edu/handle/1721.1/122682).
Master's thesis, Massachusetts Institute of Technology, 2019.
([Download](https://dspace.mit.edu/bitstream/handle/1721.1/122682/1123218324-MIT.pdf))<details><summary>BibTeX</summary><pre>
@phdthesis{swischuk2019MLandDDROM,
    title  = {Physics-based machine learning and data-driven reduced-order modeling},
    author = {Swischuk, Renee},
    year   = {2019},
    school = {Massachusetts Institute of Technology}
}</pre></details>

- \[5\] [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ)
[**Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference**](https://arxiv.org/abs/1908.11233).
arXiv:1908.11233.
([Download](https://arxiv.org/pdf/1908.11233.pdf))<details><summary>BibTeX</summary><pre>
@article{peherstorfer2019samplingMarkovian,
    title   = {Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference},
    author  = {Peherstorfer, Benjamin},
    journal = {arXiv preprint arXiv:1908.11233},
    year    = {2019}
}</pre></details>

- \[6\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Learning physics-based reduced-order models for a single-injector combustion process**](https://arc.aiaa.org/doi/10.2514/1.J058943).
_AIAA Journal_, Vol. 58:6, pp. 2658-2672, 2020.
Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Paper AIAA-2020-1411.
Also Oden Institute Report 19-13.
([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2020ROMCombustion,
    title     = {Learning physics-based reduced-order models for a single-injector combustion process},
    author    = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal   = {AIAA Journal},
    volume    = {58},
    number    = {6},
    pages     = {2658--2672},
    year      = {2020},
    publisher = {American Institute of Aeronautics and Astronautics}
}</pre></details>

- \[7\] [Qian, E.](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems**](https://www.sciencedirect.com/science/article/abs/pii/S0167278919307651).
_Physica D: Nonlinear Phenomena_, Vol. 406, May 2020, 132401.
([Download](https://kiwi.oden.utexas.edu/papers/lift-learn-scientific-machine-learning-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{QKPW2020LiftAndLearn,
    title   = {Lift \\& Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems.},
    author  = {Qian, E. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {Physica {D}: {N}onlinear {P}henomena},
    volume  = {406},
    pages   = {132401},
    url     = {https://doi.org/10.1016/j.physd.2020.132401},
    year    = {2020}
}</pre></details>

- \[8\] [Benner, P.](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), [Goyal, P.](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms**](https://arxiv.org/abs/2002.09726).
arXiv:2002.09726. Also Oden Institute Report 20-04.
([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-nonlinear-model-reduction-Benner-Goyal-Kramer-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{BGKPW2020OpInfNonPoly,
    title   = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
    author  = {Benner, P. and Goyal, P. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {arXiv preprint arXiv:2002.09726},
    year    = {2020}
}</pre></details>

- \[9\] [Yıldız, S.](https://scholar.google.com/citations?user=UVPD79MAAAAJ), [Goyal, P.](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [Benner, P.](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), and [Karasözen, B.](https://scholar.google.com/citations?user=R906kj0AAAAJ),
[**Data-driven learning of reduced-order dynamics for a parametrized shallow water equation**](https://arxiv.org/abs/2007.14079).
arXiv:2007.14079.
([Download](https://arxiv.org/pdf/2007.14079.pdf))<details><summary>BibTeX</summary><pre>
@article{SGBK2020OpInfAffine,
    title   = {Data-Driven Learning of Reduced-order Dynamics for a Parametrized Shallow Water Equation},
    author  = {Y{\i}ld{\i}z, S. and Goyal, P. and Benner, P. and Karas{\\"o}zen, B.},
    journal = {arXiv preprint arXiv:2007.14079},
    year    = {2020}
}</pre></details>

- \[10\] [McQuarrie, S. A.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[**Data-driven reduced-order models via regularized operator inference for a single-injector combustion process**](https://arxiv.org/abs/2008.02862).
arXiv:2008.02862.
([Download](https://arxiv.org/pdf/2008.02862.pdf))<details><summary>BibTeX</summary><pre>
@article{MHW2020regOpInfCombustion,
    title   = {Data-driven reduced-order models via regularized operator inference for a single-injector combustion process},
    author  = {McQuarrie, S. A. and Huang, C. and Willcox, K.},
    journal = {arXiv preprint arXiv:2008.02862},
    year    = {2020}
}</pre></details>
