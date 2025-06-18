# Literature

This page lists scholarly publications that develop, extend, or apply
Operator Inference, categorized into topics and sorted by publication year,
then by the last name of the first author. Although some could be placed in
multiple categories, each publication is only listed once.

:::{admonition} Share Your Work!
:class: hint

Don't see your publication?
[**Click here**](https://forms.gle/BgZK4b4DfuaPsGFd7)
to submit a request to add entries to this page.
:::

## Original Paper

* [**Data-driven operator inference for nonintrusive projection-based model reduction**](https://doi.org/10.1016/j.cma.2016.03.025)  
  [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ) and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2016 <details><summary>BibTeX</summary><pre>@article{peherstorfer2016opinf,
&nbsp;&nbsp;title = {Data-driven operator inference for nonintrusive projection-based model reduction},
&nbsp;&nbsp;author = {Benjamin Peherstorfer and Karen Willcox},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {306},
&nbsp;&nbsp;pages = {196--215},
&nbsp;&nbsp;year = {2016},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.cma.2016.03.025},
  }</pre></details>

## Surveys

* [**Learning physics-based models from data: Perspectives from inverse problems and model reduction**](https://doi.org/10.1017/S0962492921000064)  
  [O. Ghattas](https://scholar.google.com/citations?user=A5vhsIYAAAAJ) and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Acta Numerica, 2021 <details><summary>BibTeX</summary><pre>@article{ghattas2021acta,
&nbsp;&nbsp;title = {Learning physics-based models from data: {P}erspectives from inverse problems and model reduction},
&nbsp;&nbsp;author = {Omar Ghattas and Karen Willcox},
&nbsp;&nbsp;journal = {Acta Numerica},
&nbsp;&nbsp;volume = {30},
&nbsp;&nbsp;pages = {445--554},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;publisher = {Cambridge University Press},
&nbsp;&nbsp;doi = {10.1017/S0962492921000064},
  }</pre></details>
  <p></p>
* [**Learning nonlinear reduced models from data with operator inference**](https://doi.org/10.1146/annurev-fluid-121021-025220)  
  [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Annual Review of Fluid Mechanics, 2024 <details><summary>BibTeX</summary><pre>@article{kramer2024survey,
&nbsp;&nbsp;title = {Learning nonlinear reduced models from data with operator inference},
&nbsp;&nbsp;author = {Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
&nbsp;&nbsp;journal = {Annual Review of Fluid Mechanics},
&nbsp;&nbsp;volume = {56},
&nbsp;&nbsp;pages = {521--548},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;publisher = {Annual Reviews},
&nbsp;&nbsp;doi = {10.1146/annurev-fluid-121021-025220},
  }</pre></details>

## Methodology
### Lifting and Nonlinearity

Operator Inference learns reduced-order models with
polynomial structure. The methods developed in the following papers focus on
dealing with non-polynomial nonlinearities through variable transformations
(lifting) and/or coupling Operator Inference methods with other approximation
strategies.

* [**Transform \& Learn: A data-driven approach to nonlinear model reduction**](https://doi.org/10.2514/6.2019-3707)  
  [E. Qian](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), A. N. Marques, and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  AIAA Aviation 2019 Forum, 2019 <details><summary>BibTeX</summary><pre>@inproceedings{qian2019transform,
&nbsp;&nbsp;title = {Transform \\& {L}earn: {A} data-driven approach to nonlinear model reduction},
&nbsp;&nbsp;author = {Elizabeth Qian and Boris Kramer and Alexandre N. Marques and Karen E. Willcox},
&nbsp;&nbsp;booktitle = {AIAA Aviation 2019 Forum},
&nbsp;&nbsp;pages = {3707},
&nbsp;&nbsp;year = {2019},
&nbsp;&nbsp;doi = {10.2514/6.2019-3707},
  }</pre></details>
  <p></p>
* [**Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms**](https://doi.org/10.1016/j.cma.2020.113433)  
  [P. Benner](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), [P. Goyal](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2020 <details><summary>BibTeX</summary><pre>@article{benner2020deim,
&nbsp;&nbsp;title = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
&nbsp;&nbsp;author = {Peter Benner and Pawan Goyal and Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {372},
&nbsp;&nbsp;pages = {113433},
&nbsp;&nbsp;year = {2020},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.cma.2020.113433},
  }</pre></details>
  <p></p>
* [**Lift \& Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems**](https://doi.org/10.1016/j.physd.2020.132401)  
  [E. Qian](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Physica D: Nonlinear Phenomena, 2020 <details><summary>BibTeX</summary><pre>@article{qian2020liftandlearn,
&nbsp;&nbsp;title = {Lift \\& {L}earn: {P}hysics-informed machine learning for large-scale nonlinear dynamical systems},
&nbsp;&nbsp;author = {Elizabeth Qian and Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
&nbsp;&nbsp;journal = {Physica D: Nonlinear Phenomena},
&nbsp;&nbsp;volume = {406},
&nbsp;&nbsp;pages = {132401},
&nbsp;&nbsp;year = {2020},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.physd.2020.132401},
  }</pre></details>
  <p></p>
* [**Stability domains for quadratic-bilinear reduced-order models**](https://doi.org/10.1137/20M1364849)  
  [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  SIAM Journal on Applied Dynamical Systems, 2021 <details><summary>BibTeX</summary><pre>@article{kramer2021quadstability,
&nbsp;&nbsp;title = {Stability domains for quadratic-bilinear reduced-order models},
&nbsp;&nbsp;author = {Boris Kramer},
&nbsp;&nbsp;journal = {SIAM Journal on Applied Dynamical Systems},
&nbsp;&nbsp;volume = {20},
&nbsp;&nbsp;issue = {2},
&nbsp;&nbsp;pages = {981--996},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;publisher = {SIAM},
&nbsp;&nbsp;doi = {10.1137/20M1364849},
  }</pre></details>
  <p></p>
* [**Non-intrusive data-driven model reduction for differential algebraic equations derived from lifting transformations**](https://doi.org/10.1016/j.cma.2021.114296)  
  [P. Khodabakhshi](https://scholar.google.com/citations?user=lYr_g-MAAAAJ) and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2022 <details><summary>BibTeX</summary><pre>@article{khodabakhshi2022diffalg,
&nbsp;&nbsp;title = {Non-intrusive data-driven model reduction for differential algebraic equations derived from lifting transformations},
&nbsp;&nbsp;author = {Parisa Khodabakhshi and Karen E. Willcox},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {389},
&nbsp;&nbsp;pages = {114296},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;doi = {10.1016/j.cma.2021.114296},
  }</pre></details>
  <p></p>
* [**Reduced operator inference for nonlinear partial differential equations**](https://doi.org/10.1137/21M1393972)  
  [E. Qian](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [I. Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  SIAM Journal on Scientific Computing, 2022 <details><summary>BibTeX</summary><pre>@article{qian2022pdes,
&nbsp;&nbsp;title = {Reduced operator inference for nonlinear partial differential equations},
&nbsp;&nbsp;author = {Elizabeth Qian and Ionut-Gabriel Farcas and Karen Willcox},
&nbsp;&nbsp;journal = {SIAM Journal on Scientific Computing},
&nbsp;&nbsp;volume = {44},
&nbsp;&nbsp;issue = {4},
&nbsp;&nbsp;pages = {A1934-a1959},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;publisher = {SIAM},
&nbsp;&nbsp;doi = {10.1137/21M1393972},
  }</pre></details>
  <p></p>
* [**Exact and optimal quadratization of nonlinear finite-dimensional non-autonomous dynamical systems**](https://doi.org/10.1137/23M1561129)  
  A. Bychkov, [O. Issan](https://scholar.google.com/citations?user=eEIe19oAAAAJ), [G. Pogudin](https://scholar.google.com/citations?user=C5NP1o0AAAAJ), and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  SIAM Journal of Applied Dynamical Systems, 2024 <details><summary>BibTeX</summary><pre>@article{bychkov2024quadratization,
&nbsp;&nbsp;title = {Exact and optimal quadratization of nonlinear finite-dimensional non-autonomous dynamical systems},
&nbsp;&nbsp;author = {Andrey Bychkov and Opal Issan and Gleb Pogudin and Boris Kramer},
&nbsp;&nbsp;journal = {SIAM Journal of Applied Dynamical Systems},
&nbsp;&nbsp;volume = {23},
&nbsp;&nbsp;number = {1},
&nbsp;&nbsp;pages = {982-1016},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.1137/23M1561129},
  }</pre></details>

### Re-projection

In some cases, if the training data are chosen
judiciously, Operator Inference can recover traditional reduced-order models
defined by intrusive projection. The following papers develop and apply this
idea.

* [**Sampling low-dimensional Markovian dynamics for preasymptotically recovering reduced models from data with operator inference**](https://doi.org/10.1137/19M1292448)  
  [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ)  
  SIAM Journal on Scientific Computing, 2020 <details><summary>BibTeX</summary><pre>@article{peherstorfer2020reprojection,
&nbsp;&nbsp;title = {Sampling low-dimensional {M}arkovian dynamics for preasymptotically recovering reduced models from data with operator inference},
&nbsp;&nbsp;author = {Benjamin Peherstorfer},
&nbsp;&nbsp;journal = {SIAM Journal on Scientific Computing},
&nbsp;&nbsp;volume = {42},
&nbsp;&nbsp;issue = {5},
&nbsp;&nbsp;pages = {A3489-a3515},
&nbsp;&nbsp;year = {2020},
&nbsp;&nbsp;publisher = {SIAM},
&nbsp;&nbsp;doi = {10.1137/19M1292448},
  }</pre></details>
  <p></p>
* [**Probabilistic error estimation for non-intrusive reduced models learned from data of systems governed by linear parabolic partial differential equations**](https://doi.org/10.1051/m2an/2021010)  
  [W. I. T. Uy](https://scholar.google.com/citations?user=hNN_KRQAAAAJ) and [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ)  
  ESAIM: Mathematical Modelling and Numerical Analysis, 2021 <details><summary>BibTeX</summary><pre>@article{uy2021error,
&nbsp;&nbsp;title = {Probabilistic error estimation for non-intrusive reduced models learned from data of systems governed by linear parabolic partial differential equations},
&nbsp;&nbsp;author = {Wayne Isaac Tan Uy and Benjamin Peherstorfer},
&nbsp;&nbsp;journal = {ESAIM: Mathematical Modelling and Numerical Analysis},
&nbsp;&nbsp;volume = {55},
&nbsp;&nbsp;issue = {3},
&nbsp;&nbsp;pages = {735--761},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;publisher = {EDP Sciences},
&nbsp;&nbsp;doi = {10.1051/m2an/2021010},
  }</pre></details>
  <p></p>
* [**Exact operator inference with minimal data**](https://doi.org/10.48550/arXiv.2506.01244)  
  H. Rosenberger, B. Sanderse, and G. Stabile  
  arXiv, 2025 <details><summary>BibTeX</summary><pre>@article{rosenberger2025exactopinf,
&nbsp;&nbsp;title = {Exact operator inference with minimal data},
&nbsp;&nbsp;author = {Henrik Rosenberger and Benjamin Sanderse and Giovanni Stabile},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2506.01244},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.48550/arXiv.2506.01244},
  }</pre></details>

### Structure Preservation

The methods developed in these works augment Operator
Inference so that the resulting reduced-order models automatically inherit
certain properties from the full-order system, such as block structure,
symmetries, energy conservation, gradient structure, and more.

* [**Operator inference and physics-informed learning of low-dimensional models for incompressible flows**](https://doi.org/10.1553/etna_vol56s28)  
  [P. Benner](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), [P. Goyal](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [J. Heiland](https://scholar.google.com/citations?user=wkHSeoYAAAAJ), and [I. P. Duff](https://scholar.google.com/citations?user=OAkPFdkAAAAJ)  
  Electronic Transactions on Numerical Analysis, 2022 <details><summary>BibTeX</summary><pre>@article{benner2022incompressible,
&nbsp;&nbsp;title = {Operator inference and physics-informed learning of low-dimensional models for incompressible flows},
&nbsp;&nbsp;author = {Peter Benner and Pawan Goyal and Jan Heiland and Igor Pontes Duff},
&nbsp;&nbsp;journal = {Electronic Transactions on Numerical Analysis},
&nbsp;&nbsp;volume = {56},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;doi = {10.1553/etna_vol56s28},
  }</pre></details>
  <p></p>
* [**Hamiltonian operator inference: Physics-preserving learning of reduced-order models for canonical Hamiltonian systems**](https://doi.org/10.1016/j.physd.2021.133122)  
  [H. Sharma](https://scholar.google.com/citations?user=Pb-tL5oAAAAJ), [Z. Wang](https://scholar.google.com/citations?user=jkmwEF0AAAAJ), and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  Physica D: Nonlinear Phenomena, 2022 <details><summary>BibTeX</summary><pre>@article{sharma2022hamiltonian,
&nbsp;&nbsp;title = {Hamiltonian operator inference: {P}hysics-preserving learning of reduced-order models for canonical {H}amiltonian systems},
&nbsp;&nbsp;author = {Harsh Sharma and Zhu Wang and Boris Kramer},
&nbsp;&nbsp;journal = {Physica D: Nonlinear Phenomena},
&nbsp;&nbsp;volume = {431},
&nbsp;&nbsp;pages = {133122},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.physd.2021.133122},
  }</pre></details>
  <p></p>
* [**An operator inference oriented approach for linear mechanical systems**](https://doi.org/10.1016/j.ymssp.2023.110620)  
  Y. Filanova, [I. P. Duff](https://scholar.google.com/citations?user=OAkPFdkAAAAJ), [P. Goyal](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), and [P. Benner](https://scholar.google.com/citations?user=6zcRrC4AAAAJ)  
  Mechanical Systems and Signal Processing, 2023 <details><summary>BibTeX</summary><pre>@article{filanova2023mechanical,
&nbsp;&nbsp;title = {An operator inference oriented approach for linear mechanical systems},
&nbsp;&nbsp;author = {Yevgeniya Filanova and Igor Pontes Duff and Pawan Goyal and Peter Benner},
&nbsp;&nbsp;journal = {Mechanical Systems and Signal Processing},
&nbsp;&nbsp;volume = {200},
&nbsp;&nbsp;pages = {110620},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.ymssp.2023.110620},
  }</pre></details>
  <p></p>
* [**Canonical and noncanonical Hamiltonian operator inference**](https://doi.org/10.1016/j.cma.2023.116334)  
  [A. Gruber](https://scholar.google.com/citations?user=CJVuqfoAAAAJ) and [I. Tezaur](https://scholar.google.com/citations?user=Q3fx78kAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2023 <details><summary>BibTeX</summary><pre>@article{gruber2023hamiltonian,
&nbsp;&nbsp;title = {Canonical and noncanonical {H}amiltonian operator inference},
&nbsp;&nbsp;author = {Anthony Gruber and Irina Tezaur},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {416},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;doi = {10.1016/j.cma.2023.116334},
  }</pre></details>
  <p></p>
* [**Predicting solar wind streams from the inner-heliosphere to Earth via shifted operator inference**](https://doi.org/10.1016/j.jcp.2022.111689)  
  [O. Issan](https://scholar.google.com/citations?user=eEIe19oAAAAJ) and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  Journal of Computational Physics, 2023 <details><summary>BibTeX</summary><pre>@article{issan2023shifted,
&nbsp;&nbsp;title = {Predicting solar wind streams from the inner-heliosphere to Earth via shifted operator inference},
&nbsp;&nbsp;author = {Opal Issan and Boris Kramer},
&nbsp;&nbsp;journal = {Journal of Computational Physics},
&nbsp;&nbsp;volume = {473},
&nbsp;&nbsp;pages = {111689},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.jcp.2022.111689},
  }</pre></details>
  <p></p>
* [**Physics-informed regularization and structure preservation for learning stable reduced models from data with operator inference**](https://doi.org/10.1016/j.cma.2022.115836)  
  N. Sawant, [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), and [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2023 <details><summary>BibTeX</summary><pre>@article{sawant2023pireg,
&nbsp;&nbsp;title = {Physics-informed regularization and structure preservation for learning stable reduced models from data with operator inference},
&nbsp;&nbsp;author = {Nihar Sawant and Boris Kramer and Benjamin Peherstorfer},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {404},
&nbsp;&nbsp;pages = {115836},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.cma.2022.115836},
  }</pre></details>
  <p></p>
* [**Enforcing structure in data-driven reduced modeling through nested Operator Inference**](https://doi.org/10.1109/CDC56724.2024.10885857)  
  [N. Aretz](https://scholar.google.com/citations?user=Oje7mbAAAAAJ) and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  63rd IEEE Conference on Decision and Control (CDC), 2024 <details><summary>BibTeX</summary><pre>@inproceedings{aretz2024enforcing,
&nbsp;&nbsp;title = {Enforcing structure in data-driven reduced modeling through nested {O}perator {I}nference},
&nbsp;&nbsp;author = {Nicole Aretz and Karen Willcox},
&nbsp;&nbsp;booktitle = {63rd IEEE Conference on Decision and Control (CDC)},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;organization = {IEEE},
&nbsp;&nbsp;doi = {10.1109/CDC56724.2024.10885857},
  }</pre></details>
  <p></p>
* [**Stable sparse operator inference for nonlinear structural dynamics**](https://doi.org/10.48550/arXiv.2407.21672)  
  [P. den Boef](https://scholar.google.com/citations?user=vFlzL7kAAAAJ), [D. Manvelyan](https://scholar.google.com/citations?user=V0k8Xb4AAAAJ), [J. Maubach](https://scholar.google.com/citations?user=nBRKw6cAAAAJ), [W. Schilders](https://scholar.google.com/citations?user=UGKPyqkAAAAJ), and [N. van de Wouw](https://scholar.google.com/citations?user=pcQCbN8AAAAJ)  
  arXiv, 2024 <details><summary>BibTeX</summary><pre>@article{boef2024stablesparse,
&nbsp;&nbsp;title = {Stable sparse operator inference for nonlinear structural dynamics},
&nbsp;&nbsp;author = {Pascal {den Boef} and Diana Manvelyan and Joseph Maubach and Wil Schilders and Nathan {van de Wouw}},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2407.21672},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.48550/arXiv.2407.21672},
  }</pre></details>
  <p></p>
* [**Gradient preserving Operator Inference: Data-driven reduced-order models for equations with gradient structure**](https://doi.org/10.1016/j.cma.2024.117033)  
  [Y. Geng](https://scholar.google.com/citations?user=lms4MbwAAAAJ), [J. Singh](https://scholar.google.com/citations?user=VcmXMxgAAAAJ), [L. Ju](https://scholar.google.com/citations?user=JkKUWoAAAAAJ), [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), and [Z. Wang](https://scholar.google.com/citations?user=jkmwEF0AAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2024 <details><summary>BibTeX</summary><pre>@article{geng2024gradient,
&nbsp;&nbsp;title = {Gradient preserving {O}perator {I}nference: {D}ata-driven reduced-order models for equations with gradient structure},
&nbsp;&nbsp;author = {Yuwei Geng and Jasdeep Singh and Lili Ju and Boris Kramer and Zhu Wang},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {427},
&nbsp;&nbsp;pages = {117033},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.1016/j.cma.2024.117033},
  }</pre></details>
  <p></p>
* [**Energy-preserving reduced operator inference for efficient design and control**](https://doi.org/10.2514/6.2024-1012)  
  [T. Koike](https://scholar.google.com/citations?user=HFoIGcMAAAAJ) and [E. Qian](https://scholar.google.com/citations?user=jnHI7wQAAAAJ)  
  AIAA SciTech 2024 Forum, 2024 <details><summary>BibTeX</summary><pre>@inproceedings{koike2024energy,
&nbsp;&nbsp;title = {Energy-preserving reduced operator inference for efficient design and control},
&nbsp;&nbsp;author = {Tomoki Koike and Elizabeth Qian},
&nbsp;&nbsp;booktitle = {AIAA SciTech 2024 Forum},
&nbsp;&nbsp;pages = {1012},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.2514/6.2024-1012},
  }</pre></details>
  <p></p>
* [**Lagrangian operator inference enhanced with structure-preserving machine learning for nonintrusive model reduction of mechanical systems**](https://doi.org/10.1016/j.cma.2024.116865)  
  [H. Sharma](https://scholar.google.com/citations?user=Pb-tL5oAAAAJ), [D. A. Najera-Flores](https://scholar.google.com/citations?user=HJ-Dfl8AAAAJ), [M. D. Todd](https://scholar.google.com/citations?user=jzY8TSkAAAAJ), and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2024 <details><summary>BibTeX</summary><pre>@article{sharma2024lagrangian,
&nbsp;&nbsp;title = {Lagrangian operator inference enhanced with structure-preserving machine learning for nonintrusive model reduction of mechanical systems},
&nbsp;&nbsp;author = {Harsh Sharma and David A Najera-Flores and Michael D Todd and Boris Kramer},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {423},
&nbsp;&nbsp;pages = {116865},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.cma.2024.116865},
  }</pre></details>
  <p></p>
* [**Preserving Lagrangian structure in data-driven reduced-order modeling of large-scale mechanical systems**](https://doi.org/10.1016/j.physd.2024.134128)  
  [H. Sharma](https://scholar.google.com/citations?user=Pb-tL5oAAAAJ) and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  Physica D: Nonlinear Phenomena, 2024 <details><summary>BibTeX</summary><pre>@article{sharma2024preserving,
&nbsp;&nbsp;title = {Preserving {L}agrangian structure in data-driven reduced-order modeling of large-scale mechanical systems},
&nbsp;&nbsp;author = {Harsh Sharma and Boris Kramer},
&nbsp;&nbsp;journal = {Physica D: Nonlinear Phenomena},
&nbsp;&nbsp;volume = {462},
&nbsp;&nbsp;pages = {134128},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.1016/j.physd.2024.134128},
  }</pre></details>
  <p></p>
* [**Data-driven reduced-order models for port-Hamiltonian systems with Operator Inference**](https://doi.org/10.48550/arXiv.2501.02183)  
  [Y. Geng](https://scholar.google.com/citations?user=lms4MbwAAAAJ), [L. Ju](https://scholar.google.com/citations?user=JkKUWoAAAAAJ), [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), and [Z. Wang](https://scholar.google.com/citations?user=jkmwEF0AAAAJ)  
  arXiv, 2025 <details><summary>BibTeX</summary><pre>@article{geng2025porthamiltonian,
&nbsp;&nbsp;title = {Data-driven reduced-order models for port-{H}amiltonian systems with {O}perator {I}nference},
&nbsp;&nbsp;author = {Yuwei Geng and Lili Ju and Boris Kramer and Zhu Wang},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2501.02183},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.48550/arXiv.2501.02183},
  }</pre></details>
  <p></p>
* [**Variationally consistent Hamiltonian model reduction**](https://doi.org/10.1137/24M1652490)  
  [A. Gruber](https://scholar.google.com/citations?user=CJVuqfoAAAAJ) and [I. Tezaur](https://scholar.google.com/citations?user=Q3fx78kAAAAJ)  
  SIAM Journal on Applied Dynamical Systems, 2025 <details><summary>BibTeX</summary><pre>@article{gruber2025variational,
&nbsp;&nbsp;author = {Anthony Gruber and Irina Tezaur},
&nbsp;&nbsp;title = {Variationally consistent {H}amiltonian model reduction},
&nbsp;&nbsp;journal = {SIAM Journal on Applied Dynamical Systems},
&nbsp;&nbsp;volume = {24},
&nbsp;&nbsp;number = {1},
&nbsp;&nbsp;pages = {376-414},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.1137/24M1652490},
  }</pre></details>
  <p></p>
* [**Physically consistent predictive reduced-order modeling by enhancing Operator Inference with state constraints**](https://doi.org/10.48550/arXiv.2502.03672)  
  [H. Kim](https://scholar.google.com/citations?user=sdR-LZ4AAAAJ) and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  arXiv, 2025 <details><summary>BibTeX</summary><pre>@article{kim2025stateconstraints,
&nbsp;&nbsp;title = {Physically consistent predictive reduced-order modeling by enhancing {O}perator {I}nference with state constraints},
&nbsp;&nbsp;author = {Hyeonghun Kim and Boris Kramer},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2502.03672},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.48550/arXiv.2502.03672},
  }</pre></details>

### Parametric Problems

Many systems depend on independent parameters that
describe material properties or other physical characteristics of the
phenomenon being modeled. The following papers develop Operator Inference
approaches that are specifically designed for parametric problems.

* [**Learning reduced-order dynamics for parametrized shallow water equations from data**](https://doi.org/10.1002/fld.4998)  
  [S. Yıldız](https://scholar.google.com/citations?user=UVPD79MAAAAJ), [P. Goyal](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [P. Benner](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), and [B. Karasözen](https://scholar.google.com/citations?user=R906kj0AAAAJ)  
  International Journal for Numerical Methods in Fluids, 2021 <details><summary>BibTeX</summary><pre>@article{yildiz2021shallow,
&nbsp;&nbsp;title = {Learning reduced-order dynamics for parametrized shallow water equations from data},
&nbsp;&nbsp;author = {S\\\\"{u}leyman Y\\i{}ld\\i{}z and Pawan Goyal and Peter Benner and B\\\\"{u}lent Karas\\\\"{o}zen},
&nbsp;&nbsp;journal = {International Journal for Numerical Methods in Fluids},
&nbsp;&nbsp;volume = {93},
&nbsp;&nbsp;issue = {8},
&nbsp;&nbsp;pages = {2803--2821},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;publisher = {Wiley Online Library},
&nbsp;&nbsp;doi = {10.1002/fld.4998},
  }</pre></details>
  <p></p>
* [**Parametric non-intrusive reduced-order models via operator inference for large-scale rotating detonation engine simulations**](https://doi.org/10.2514/6.2023-0172)  
  [I. Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), R. Gundevia, R. Munipalli, and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  AIAA SciTech 2023 Forum, 2023 <details><summary>BibTeX</summary><pre>@inproceedings{farcas2023parametric,
&nbsp;&nbsp;title = {Parametric non-intrusive reduced-order models via operator inference for large-scale rotating detonation engine simulations},
&nbsp;&nbsp;author = {Ionut-Gabriel Farcas and Rayomand Gundevia and Ramakanth Munipalli and Karen E. Willcox},
&nbsp;&nbsp;booktitle = {AIAA SciTech 2023 Forum},
&nbsp;&nbsp;pages = {0172},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;doi = {10.2514/6.2023-0172},
  }</pre></details>
  <p></p>
* [**Nonintrusive reduced-order models for parametric partial differential equations via data-driven operator inference**](https://doi.org/10.1137/21M1452810)  
  [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [P. Khodabakhshi](https://scholar.google.com/citations?user=lYr_g-MAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  SIAM Journal on Scientific Computing, 2023 <details><summary>BibTeX</summary><pre>@article{mcquarrie2023parametric,
&nbsp;&nbsp;title = {Nonintrusive reduced-order models for parametric partial differential equations via data-driven operator inference},
&nbsp;&nbsp;author = {Shane A McQuarrie and Parisa Khodabakhshi and Karen Willcox},
&nbsp;&nbsp;journal = {SIAM Journal on Scientific Computing},
&nbsp;&nbsp;volume = {45},
&nbsp;&nbsp;issue = {4},
&nbsp;&nbsp;pages = {A1917-A1946},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {SIAM},
&nbsp;&nbsp;doi = {10.1137/21M1452810},
  }</pre></details>
  <p></p>
* [**Tensor parametric Hamiltonian operator inference**](https://doi.org/10.48550/arXiv.2502.10888)  
  [A. Vijaywargiya](https://scholar.google.com/citations?user=_fcSwDYAAAAJ), [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), and [A. Gruber](https://scholar.google.com/citations?user=CJVuqfoAAAAJ)  
  arXiv, 2025 <details><summary>BibTeX</summary><pre>@article{vijaywargiya2025tensoropinf,
&nbsp;&nbsp;title = {Tensor parametric {H}amiltonian operator inference},
&nbsp;&nbsp;author = {Arjun Vijaywargiya and Shane A. McQuarrie and Anthony Gruber},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2502.10888},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.48550/arXiv.2502.10888},
  }</pre></details>

### Statistical Methods

These papers focus on problems with noisy or missing
data, stochastic systems, and methods for constructing probabilistic
reduced-order models with Operator Inference.

* [**Operator inference of non-Markovian terms for learning reduced models from partially observed state trajectories**](https://doi.org/10.1007/s10915-021-01580-2)  
  [W. I. T. Uy](https://scholar.google.com/citations?user=hNN_KRQAAAAJ) and [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ)  
  Journal of Scientific Computing, 2021 <details><summary>BibTeX</summary><pre>@article{uy2021partial,
&nbsp;&nbsp;title = {Operator inference of non-{M}arkovian terms for learning reduced models from partially observed state trajectories},
&nbsp;&nbsp;author = {Wayne Isaac Tan Uy and Benjamin Peherstorfer},
&nbsp;&nbsp;journal = {Journal of Scientific Computing},
&nbsp;&nbsp;volume = {88},
&nbsp;&nbsp;issue = {3},
&nbsp;&nbsp;pages = {1--31},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;publisher = {Springer},
&nbsp;&nbsp;doi = {10.1007/s10915-021-01580-2},
  }</pre></details>
  <p></p>
* [**Bayesian operator inference for data-driven reduced-order modeling**](https://doi.org/10.1016/j.cma.2022.115336)  
  [M. Guo](https://scholar.google.com/citations?user=eON6MykAAAAJ), [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2022 <details><summary>BibTeX</summary><pre>@article{guo2022bayesopinf,
&nbsp;&nbsp;title = {Bayesian operator inference for data-driven reduced-order modeling},
&nbsp;&nbsp;author = {Mengwu Guo and Shane A McQuarrie and Karen Willcox},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {402},
&nbsp;&nbsp;pages = {115336},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.cma.2022.115336},
  }</pre></details>
  <p></p>
* [**Active operator inference for learning low-dimensional dynamical-system models from noisy data**](https://doi.org/10.1137/21M1439729)  
  [W. I. T. Uy](https://scholar.google.com/citations?user=hNN_KRQAAAAJ), Y. Wang, [Y. Wen](https://scholar.google.com/citations?user=uXJoQCAAAAAJ), and [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ)  
  SIAM Journal on Scientific Computing, 2023 <details><summary>BibTeX</summary><pre>@article{uy2023active,
&nbsp;&nbsp;title = {Active operator inference for learning low-dimensional dynamical-system models from noisy data},
&nbsp;&nbsp;author = {Wayne Isaac Tan Uy and Yuepeng Wang and Yuxiao Wen and Benjamin Peherstorfer},
&nbsp;&nbsp;journal = {SIAM Journal on Scientific Computing},
&nbsp;&nbsp;volume = {45},
&nbsp;&nbsp;issue = {4},
&nbsp;&nbsp;pages = {A1462-a1490},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {SIAM},
&nbsp;&nbsp;doi = {10.1137/21M1439729},
  }</pre></details>
  <p></p>
* [**Operator inference with roll outs for learning reduced models from scarce and low-quality data**](https://doi.org/10.1016/j.camwa.2023.06.012)  
  [W. I. T. Uy](https://scholar.google.com/citations?user=hNN_KRQAAAAJ), [D. Hartmann](https://scholar.google.com/citations?user=4XvBneEAAAAJ), and [B. Peherstorfer](https://scholar.google.com/citations?user=C81WhlkAAAAJ)  
  Computers \& Mathematics with Applications, 2023 <details><summary>BibTeX</summary><pre>@article{uy2023rollouts,
&nbsp;&nbsp;title = {Operator inference with roll outs for learning reduced models from scarce and low-quality data},
&nbsp;&nbsp;author = {Wayne Isaac Tan Uy and Dirk Hartmann and Benjamin Peherstorfer},
&nbsp;&nbsp;journal = {Computers \\& Mathematics with Applications},
&nbsp;&nbsp;volume = {145},
&nbsp;&nbsp;pages = {224--239},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.camwa.2023.06.012},
  }</pre></details>
  <p></p>
* [**Learning stochastic reduced models from data: a nonintrusive approach**](https://doi.org/10.48550/arXiv.2407.05724)  
  [M. A. Freitag](https://scholar.google.com/citations?user=iE4t4WcAAAAJ), [J. M. Nicolaus](https://scholar.google.com/citations?user=47DRMUwAAAAJ), and M. Redmann  
  arXiv, 2024 <details><summary>BibTeX</summary><pre>@misc{freitag2024stochastic,
&nbsp;&nbsp;title = {Learning stochastic reduced models from data: a nonintrusive approach},
&nbsp;&nbsp;author = {Melina A. Freitag and Jan Martin Nicolaus and Martin Redmann},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2407.05724},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.48550/arXiv.2407.05724},
  }</pre></details>
  <p></p>
* [**Bayesian learning with Gaussian processes for low-dimensional representations of time-dependent nonlinear systems**](https://doi.org/10.1016/j.physd.2025.134572)  
  [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [A. Chaudhuri](https://scholar.google.com/citations?user=oGL9YJIAAAAJ), [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ), and [M. Guo](https://scholar.google.com/citations?user=eON6MykAAAAJ)  
  Physica D: Nonlinear Phenomena, 2025 <details><summary>BibTeX</summary><pre>@article{mcquarrie2025gpbayesopinf,
&nbsp;&nbsp;title = {Bayesian learning with {G}aussian processes for low-dimensional representations of time-dependent nonlinear systems},
&nbsp;&nbsp;author = {Shane A. McQuarrie and Anirban Chaudhuri and Karen E. Willcox and Mengwu Guo},
&nbsp;&nbsp;journal = {Physica D: Nonlinear Phenomena},
&nbsp;&nbsp;volume = {475},
&nbsp;&nbsp;pages = {134572},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.1016/j.physd.2025.134572},
  }</pre></details>

### Domain Decomposition

The methods in the following papers focus on scalability
and accuracy improvements by decomposition spatial or latent space domains and
learning a coupled system of reduced-order models.

* [**Localized non-intrusive reduced-order modelling in the operator inference framework**](https://doi.org/10.1098/rsta.2021.0206)  
  [R. Geelen](https://scholar.google.com/citations?user=vBzKRMsAAAAJ) and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Philosophical Transactions of the Royal Society A, 2022 <details><summary>BibTeX</summary><pre>@article{geelen2022localized,
&nbsp;&nbsp;title = {Localized non-intrusive reduced-order modelling in the operator inference framework},
&nbsp;&nbsp;author = {Rudy Geelen and Karen Willcox},
&nbsp;&nbsp;journal = {Philosophical Transactions of the Royal Society A},
&nbsp;&nbsp;volume = {380},
&nbsp;&nbsp;number = {2229},
&nbsp;&nbsp;pages = {20210206},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;doi = {10.1098/rsta.2021.0206},
  }</pre></details>
  <p></p>
* [**Domain decomposition for data-driven reduced modeling of large-scale systems**](https://doi.org/10.2514/1.J063715)  
  [I. Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), R. P. Gundevia, R. Munipalli, and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  AIAA Journal, 2024 <details><summary>BibTeX</summary><pre>@article{farcas2024domaindecomposition,
&nbsp;&nbsp;title = {Domain decomposition for data-driven reduced modeling of large-scale systems},
&nbsp;&nbsp;author = {Ionut-Gabriel Farcas and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
&nbsp;&nbsp;journal = {AIAA Journal},
&nbsp;&nbsp;volume = {62},
&nbsp;&nbsp;number = {11},
&nbsp;&nbsp;pages = {4071-4086},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.2514/1.J063715},
  }</pre></details>
  <p></p>
* [**Non-intrusive reduced-order modeling for dynamical systems with spatially localized features**](https://doi.org/10.1016/j.cma.2025.118115)  
  [L. Gkimisis](https://scholar.google.com/citations?user=0GzUUzMAAAAJ), [N. Aretz](https://scholar.google.com/citations?user=Oje7mbAAAAAJ), [M. Tezzele](https://scholar.google.com/citations?user=UPcyNXIAAAAJ), [T. Richter](https://scholar.google.com/citations?user=C8R6xtMAAAAJ), [P. Benner](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2025 <details><summary>BibTeX</summary><pre>@article{gkimisis2025spatiallylocal,
&nbsp;&nbsp;title = {Non-intrusive reduced-order modeling for dynamical systems with spatially localized features},
&nbsp;&nbsp;author = {Leonidas Gkimisis and Nicole Aretz and Marco Tezzele and Thomas Richter and Peter Benner and Karen E. Willcox},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {444},
&nbsp;&nbsp;pages = {118115},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.1016/j.cma.2025.118115},
  }</pre></details>

### Nonlinear Manifolds

Traditional model reduction methods approximate the
high-dimensional system state with a low-dimensional linear (or affine)
representation. The methods in these papers explore using nonlinear
low-dimensional representations in the context of Operator Inference.

* [**A quadratic decoder approach to nonintrusive reduced-order modeling of nonlinear dynamical systems**](https://doi.org/10.1002/pamm.202200049)  
  [P. Benner](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), [P. Goyal](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [J. Heiland](https://scholar.google.com/citations?user=wkHSeoYAAAAJ), and [I. P. Duff](https://scholar.google.com/citations?user=OAkPFdkAAAAJ)  
  Proceedings in Applied Mathematics and Mechanics, 2023 <details><summary>BibTeX</summary><pre>@article{benner2023quaddecoder,
&nbsp;&nbsp;title = {A quadratic decoder approach to nonintrusive reduced-order modeling of nonlinear dynamical systems},
&nbsp;&nbsp;author = {Peter Benner and Pawan Goyal and Jan Heiland and Igor Pontes Duff},
&nbsp;&nbsp;journal = {Proceedings in Applied Mathematics and Mechanics},
&nbsp;&nbsp;volume = {23},
&nbsp;&nbsp;number = {1},
&nbsp;&nbsp;pages = {e202200049},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;doi = {10.1002/pamm.202200049},
  }</pre></details>
  <p></p>
* [**Operator inference for non-intrusive model reduction with quadratic manifolds**](https://doi.org/10.1016/j.cma.2022.115717)  
  [R. Geelen](https://scholar.google.com/citations?user=vBzKRMsAAAAJ), [S. Wright](https://scholar.google.com/citations?user=VFQRIOwAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Methods in Applied Mechanics and Engineering, 2023 <details><summary>BibTeX</summary><pre>@article{geelen2023quadmanifold,
&nbsp;&nbsp;title = {Operator inference for non-intrusive model reduction with quadratic manifolds},
&nbsp;&nbsp;author = {Rudy Geelen and Stephen Wright and Karen Willcox},
&nbsp;&nbsp;journal = {Computer Methods in Applied Mechanics and Engineering},
&nbsp;&nbsp;volume = {403},
&nbsp;&nbsp;pages = {115717},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.cma.2022.115717},
  }</pre></details>
  <p></p>
* [**Learning latent representations in high-dimensional state spaces using polynomial manifold constructions**](https://doi.org/10.1109/CDC49753.2023.10384209)  
  [R. Geelen](https://scholar.google.com/citations?user=vBzKRMsAAAAJ), [L. Balzano](https://scholar.google.com/citations?user=X6fRNfUAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  62nd IEEE Conference on Decision and Control (CDC), 2023 <details><summary>BibTeX</summary><pre>@inproceedings{geelen2023latent,
&nbsp;&nbsp;author = {Rudy Geelen and Laura Balzano and Karen Willcox},
&nbsp;&nbsp;booktitle = {62nd IEEE Conference on Decision and Control (CDC)},
&nbsp;&nbsp;title = {Learning latent representations in high-dimensional state spaces using polynomial manifold constructions},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;volume = {},
&nbsp;&nbsp;number = {},
&nbsp;&nbsp;pages = {4960-4965},
&nbsp;&nbsp;doi = {10.1109/CDC49753.2023.10384209},
  }</pre></details>
  <p></p>
* [**Learning physics-based reduced-order models from data using nonlinear manifolds**](https://doi.org/10.1063/5.0170105)  
  [R. Geelen](https://scholar.google.com/citations?user=vBzKRMsAAAAJ), [L. Balzano](https://scholar.google.com/citations?user=X6fRNfUAAAAJ), [S. Wright](https://scholar.google.com/citations?user=VFQRIOwAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Chaos: An Interdisciplinary Journal of Nonlinear Science, 2024 <details><summary>BibTeX</summary><pre>@article{geelen2024nonlinmanifold,
&nbsp;&nbsp;title = {Learning physics-based reduced-order models from data using nonlinear manifolds},
&nbsp;&nbsp;author = {Rudy Geelen and Laura Balzano and Stephen Wright and Karen Willcox},
&nbsp;&nbsp;journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
&nbsp;&nbsp;volume = {34},
&nbsp;&nbsp;number = {3},
&nbsp;&nbsp;pages = {033122},
&nbsp;&nbsp;year = {2024},
&nbsp;&nbsp;doi = {10.1063/5.0170105},
  }</pre></details>

### Scalability

These works focus on the computational challenge of applying
Operator Inference to large-scale problems.

* [**Data-driven reduced-order models via regularised operator inference for a single-injector combustion process**](https://doi.org/10.1080/03036758.2020.1863237)  
  [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [C. Huang](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Journal of the Royal Society of New Zealand, 2021 <details><summary>BibTeX</summary><pre>@article{mcquarrie2021combustion,
&nbsp;&nbsp;title = {Data-driven reduced-order models via regularised operator inference for a single-injector combustion process},
&nbsp;&nbsp;author = {Shane A McQuarrie and Cheng Huang and Karen Willcox},
&nbsp;&nbsp;journal = {Journal of the Royal Society of New Zealand},
&nbsp;&nbsp;volume = {51},
&nbsp;&nbsp;issue = {2},
&nbsp;&nbsp;pages = {194-211},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;publisher = {Taylor \\& Francis},
&nbsp;&nbsp;doi = {10.1080/03036758.2020.1863237},
  }</pre></details>
  <p></p>
* [**A parallel implementation of reduced-order modeling of large-scale systems**](https://doi.org/10.2514/6.2025-1170)  
  [I. Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), R. P. Gundevia, R. Munipalli, and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  AIAA SciTech 2025 Forum, 2025 <details><summary>BibTeX</summary><pre>@inbook{farcas2025parallel,
&nbsp;&nbsp;title = {A parallel implementation of reduced-order modeling of large-scale systems},
&nbsp;&nbsp;author = {Ionut-Gabriel Farcas and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
&nbsp;&nbsp;booktitle = {AIAA SciTech 2025 Forum},
&nbsp;&nbsp;pages = {1170},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.2514/6.2025-1170},
  }</pre></details>
  <p></p>
* [**Distributed computing for physics-based data-driven reduced modeling at scale: Application to a rotating detonation rocket engine**](https://doi.org/10.1016/j.cpc.2025.109619)  
  I. Farcaş, R. P. Gundevia, R. Munipalli, and [K. E. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  Computer Physics Communications, 2025 <details><summary>BibTeX</summary><pre>@article{farcas2025distributed,
&nbsp;&nbsp;title = {Distributed computing for physics-based data-driven reduced modeling at scale: {A}pplication to a rotating detonation rocket engine},
&nbsp;&nbsp;author = {Ionut-Gabriel Farcaş and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
&nbsp;&nbsp;journal = {Computer Physics Communications},
&nbsp;&nbsp;volume = {313},
&nbsp;&nbsp;pages = {109619},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.1016/j.cpc.2025.109619},
  }</pre></details>

## Applications

* [**Learning physics-based reduced-order models for a single-injector combustion process**](https://doi.org/10.2514/1.J058943)  
  [R. Swischuk](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ), [C. Huang](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ)  
  AIAA Journal, 2020 <details><summary>BibTeX</summary><pre>@article{swischuk2020combustion,
&nbsp;&nbsp;title = {Learning physics-based reduced-order models for a single-injector combustion process},
&nbsp;&nbsp;author = {Renee Swischuk and Boris Kramer and Cheng Huang and Karen Willcox},
&nbsp;&nbsp;journal = {AIAA Journal},
&nbsp;&nbsp;volume = {58},
&nbsp;&nbsp;issue = {6},
&nbsp;&nbsp;pages = {2658--2672},
&nbsp;&nbsp;year = {2020},
&nbsp;&nbsp;publisher = {American Institute of Aeronautics and Astronautics},
&nbsp;&nbsp;doi = {10.2514/1.J058943},
  }</pre></details>
  <p></p>
* [**Performance comparison of data-driven reduced models for a single-injector combustion process**](https://doi.org/10.2514/6.2021-3633)  
  P. Jain, [S. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  AIAA Propulsion and Energy 2021 Forum, 2021 <details><summary>BibTeX</summary><pre>@inproceedings{jain2021performance,
&nbsp;&nbsp;title = {Performance comparison of data-driven reduced models for a single-injector combustion process},
&nbsp;&nbsp;author = {Parikshit Jain and Shane McQuarrie and Boris Kramer},
&nbsp;&nbsp;booktitle = {AIAA Propulsion and Energy 2021 Forum},
&nbsp;&nbsp;pages = {3633},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;doi = {10.2514/6.2021-3633},
  }</pre></details>
  <p></p>
* [**Non-Intrusive reduced models based on operator inference for chaotic systems**](https://arxiv.org/abs/2206.01604)  
  J. L. d. S. Almeida, [A. C. Pires](https://scholar.google.com/citations?user=qIUw-GEAAAAJ), K. F. V. Cid, and [A. C. Nogueira Jr.](https://scholar.google.com/citations?user=66DEy5wAAAAJ)  
  arXiv, 2022 <details><summary>BibTeX</summary><pre>@article{almeida2022chaotic,
&nbsp;&nbsp;title = {Non-Intrusive reduced models based on operator inference for chaotic systems},
&nbsp;&nbsp;author = {Jo\\~{a}o Lucas de Sousa Almeida and Arthur Cancellieri Pires and Klaus Feine Vaz Cid and Alberto Costa Nogueira Jr},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2206.01604},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;url = {https://arxiv.org/abs/2206.01604},
  }</pre></details>
  <p></p>
* [**Data-driven reduced-order model for atmospheric CO2 dispersion**](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/aaaifss2022/2/paper.pdf)  
  P. R. B. Rocha, [M. S. d. P. Gomes](https://scholar.google.com/citations?user=s6mocWAAAAAJ), J. L. d. S. Almeida, A. M. Carvalho, and [A. C. Nogueira Jr.](https://scholar.google.com/citations?user=66DEy5wAAAAJ)  
  AAAI Fall Symposium, 2022 <details><summary>BibTeX</summary><pre>@inproceedings{rocha2022c02,
&nbsp;&nbsp;title = {Data-driven reduced-order model for atmospheric {CO}2 dispersion},
&nbsp;&nbsp;author = {Pedro Roberto Barbosa Rocha and Marcos Sebasti\\~{a}o de Paula Gomes and Jo\\~{a}o Lucas de Sousa Almeida and Allan M Carvalho and Alberto Costa Nogueira Jr},
&nbsp;&nbsp;booktitle = {AAAI Fall Symposium},
&nbsp;&nbsp;year = {2022},
&nbsp;&nbsp;url = {https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/aaaifss2022/2/paper.pdf},
  }</pre></details>
  <p></p>
* [**Reduced-order modeling of the two-dimensional Rayleigh--Bénard convection flow through a non-intrusive operator inference**](https://doi.org/10.1016/j.engappai.2023.106923)  
  P. R. B. Rocha, J. L. d. S. Almeida, [M. S. d. P. Gomes](https://scholar.google.com/citations?user=s6mocWAAAAAJ), and [A. C. Nogueira Jr.](https://scholar.google.com/citations?user=66DEy5wAAAAJ)  
  Engineering Applications of Artificial Intelligence, 2023 <details><summary>BibTeX</summary><pre>@article{rocha2023convection,
&nbsp;&nbsp;title = {Reduced-order modeling of the two-dimensional {R}ayleigh--{B}\\'{e}nard convection flow through a non-intrusive operator inference},
&nbsp;&nbsp;author = {Pedro Roberto Barbosa Rocha and Jo\\~{a}o Lucas de Sousa Almeida and Marcos Sebasti\\~{a}o de Paula Gomes and Alberto Costa Nogueira Jr},
&nbsp;&nbsp;journal = {Engineering Applications of Artificial Intelligence},
&nbsp;&nbsp;volume = {126},
&nbsp;&nbsp;pages = {106923},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;publisher = {Elsevier},
&nbsp;&nbsp;doi = {10.1016/j.engappai.2023.106923},
  }</pre></details>
  <p></p>
* [**Data-driven model reduction via operator inference for coupled aeroelastic flutter**](https://doi.org/10.2514/6.2023-0330)  
  [B. G. Zastrow](https://scholar.google.com/citations?user=ODLjrBAAAAAJ), [A. Chaudhuri](https://scholar.google.com/citations?user=oGL9YJIAAAAJ), [K. Willcox](https://scholar.google.com/citations?user=axvGyXoAAAAJ), [A. S. Ashley](https://scholar.google.com/citations?user=9KFAXLYAAAAJ), and M. C. Henson  
  AIAA SciTech 2023 Forum, 2023 <details><summary>BibTeX</summary><pre>@inproceedings{zastrow2023flutter,
&nbsp;&nbsp;title = {Data-driven model reduction via operator inference for coupled aeroelastic flutter},
&nbsp;&nbsp;author = {Benjamin G Zastrow and Anirban Chaudhuri and Karen Willcox and Anthony S Ashley and Michael C Henson},
&nbsp;&nbsp;booktitle = {AIAA SciTech 2023 Forum},
&nbsp;&nbsp;pages = {0330},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;doi = {10.2514/6.2023-0330},
  }</pre></details>
  <p></p>
* [**Operator inference-based model order reduction of thermal protection system finite element simulations**](https://doi.org/10.2514/6.2025-2133)  
  [P. J. Blonigan](https://scholar.google.com/citations?user=lOmH5XcAAAAJ), [J. Tencer](https://scholar.google.com/citations?user=M6AwtC4AAAAJ), [S. Babiniec](https://scholar.google.com/citations?user=xcSVh00AAAAJ), and [J. Murray](https://scholar.google.com/citations?user=NScAg7AAAAAJ)  
  AIAA SciTech 2025 Forum, 2025 <details><summary>BibTeX</summary><pre>@inbook{blonigan2025thermal,
&nbsp;&nbsp;title = {Operator inference-based model order reduction of thermal protection system finite element simulations},
&nbsp;&nbsp;author = {Patrick J. Blonigan and John Tencer and Sean Babiniec and Jonathan Murray},
&nbsp;&nbsp;booktitle = {AIAA SciTech 2025 Forum},
&nbsp;&nbsp;pages = {2133},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.2514/6.2025-2133},
  }</pre></details>
  <p></p>
* [**Parametric Operator Inference to simulate the purging process in semiconductor manufacturing**](https://doi.org/10.48550/arXiv.2504.03990)  
  S. Kang, [H. Kim](https://scholar.google.com/citations?user=sdR-LZ4AAAAJ), and [B. Kramer](https://scholar.google.com/citations?user=yfmbPNoAAAAJ)  
  arXiv, 2025 <details><summary>BibTeX</summary><pre>@article{kang2025semiconductor,
&nbsp;&nbsp;title = {Parametric {O}perator {I}nference to simulate the purging process in semiconductor manufacturing},
&nbsp;&nbsp;author = {Seunghyon Kang and Hyeonghun Kim and Boris Kramer},
&nbsp;&nbsp;journal = {arXiv},
&nbsp;&nbsp;volume = {2504.03990},
&nbsp;&nbsp;year = {2025},
&nbsp;&nbsp;doi = {10.48550/arXiv.2504.03990},
  }</pre></details>

## Dissertations and Theses

* [**Physics-based machine learning and data-driven reduced-order modeling**](https://dspace.mit.edu/handle/1721.1/122682)  
  [R. C. Swischuk](https://scholar.google.com/citations?user=L9D0LBsAAAAJ)  
  Master's Thesis, Massachusetts Institute of Technology, 2019 <details><summary>BibTeX</summary><pre>@mastersthesis{swischuk2019thesis,
&nbsp;&nbsp;title = {Physics-based machine learning and data-driven reduced-order modeling},
&nbsp;&nbsp;author = {Renee Copland Swischuk},
&nbsp;&nbsp;school = {Massachusetts Institute of Technology},
&nbsp;&nbsp;year = {2019},
&nbsp;&nbsp;url = {https://dspace.mit.edu/handle/1721.1/122682},
  }</pre></details>
  <p></p>
* [**A scientific machine learning approach to learning reduced models for nonlinear partial differential equations**](https://dspace.mit.edu/handle/1721.1/130748)  
  [E. Y. Qian](https://scholar.google.com/citations?user=jnHI7wQAAAAJ)  
  PhD Thesis, Massachusetts Institute of Technology, 2021 <details><summary>BibTeX</summary><pre>@phdthesis{qian2021thesis,
&nbsp;&nbsp;title = {A scientific machine learning approach to learning reduced models for nonlinear partial differential equations},
&nbsp;&nbsp;author = {Elizabeth Yi Qian},
&nbsp;&nbsp;school = {Massachusetts Institute of Technology},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;url = {https://dspace.mit.edu/handle/1721.1/130748},
  }</pre></details>
  <p></p>
* [**Toward predictive digital twins for self-aware unmanned aerial vehicles: Non-intrusive reduced order models and experimental data analysis**](http://dx.doi.org/10.26153/tsw/14557)  
  S. J. Salinger  
  Master's Thesis, The University of Texas at Austin, 2021 <details><summary>BibTeX</summary><pre>@mastersthesis{salinger2021thesis,
&nbsp;&nbsp;title = {Toward predictive digital twins for self-aware unmanned aerial vehicles: {N}on-intrusive reduced order models and experimental data analysis},
&nbsp;&nbsp;author = {Stephanie Joyce Salinger},
&nbsp;&nbsp;school = {The University of Texas at Austin},
&nbsp;&nbsp;year = {2021},
&nbsp;&nbsp;url = {http://dx.doi.org/10.26153/tsw/14557},
  }</pre></details>
  <p></p>
* [**Data-driven parametric reduced-order models: Operator inference for reactive flow applications**](https://doi.org/10.26153/tsw/50172)  
  [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ)  
  PhD Thesis, The University of Texas at Austin, 2023 <details><summary>BibTeX</summary><pre>@phdthesis{mcquarrie2023thesis,
&nbsp;&nbsp;title = {Data-driven parametric reduced-order models: {O}perator inference for reactive flow applications},
&nbsp;&nbsp;author = {Shane Alexander McQuarrie},
&nbsp;&nbsp;school = {The University of Texas at Austin},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;doi = {10.26153/tsw/50172},
  }</pre></details>
  <p></p>
* **Learning structured and stable reduced models from data with operator inference**  
  N. Sawant  
  PhD Thesis, New York University, 2023 <details><summary>BibTeX</summary><pre>@phdthesis{sawant2023thesis,
&nbsp;&nbsp;title = {Learning structured and stable reduced models from data with operator inference},
&nbsp;&nbsp;author = {Nihar Sawant},
&nbsp;&nbsp;year = {2023},
&nbsp;&nbsp;school = {New York University},
  }</pre></details>
## BibTex File

:::{admonition} Sorted alphabetically by author
:class: dropdown seealso

```bibtex
@article{almeida2022chaotic,
    title = {Non-Intrusive reduced models based on operator inference for chaotic systems},
    author = {Jo\~{a}o Lucas de Sousa Almeida and Arthur Cancellieri Pires and Klaus Feine Vaz Cid and Alberto Costa Nogueira Jr},
    journal = {arXiv},
    volume = {2206.01604},
    year = {2022},
}

@inproceedings{aretz2024enforcing,
    title = {Enforcing structure in data-driven reduced modeling through nested {O}perator {I}nference},
    author = {Nicole Aretz and Karen Willcox},
    booktitle = {63rd IEEE Conference on Decision and Control (CDC)},
    year = {2024},
    organization = {IEEE},
    doi = {10.1109/CDC56724.2024.10885857},
}

@article{benner2020deim,
    title = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
    author = {Peter Benner and Pawan Goyal and Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {372},
    pages = {113433},
    year = {2020},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2020.113433},
}

@article{benner2022incompressible,
    title = {Operator inference and physics-informed learning of low-dimensional models for incompressible flows},
    author = {Peter Benner and Pawan Goyal and Jan Heiland and Igor Pontes Duff},
    journal = {Electronic Transactions on Numerical Analysis},
    volume = {56},
    year = {2022},
    doi = {10.1553/etna_vol56s28},
}

@article{benner2023quaddecoder,
    title = {A quadratic decoder approach to nonintrusive reduced-order modeling of nonlinear dynamical systems},
    author = {Peter Benner and Pawan Goyal and Jan Heiland and Igor Pontes Duff},
    journal = {Proceedings in Applied Mathematics and Mechanics},
    volume = {23},
    number = {1},
    pages = {e202200049},
    year = {2023},
    doi = {10.1002/pamm.202200049},
}

@inbook{blonigan2025thermal,
    title = {Operator inference-based model order reduction of thermal protection system finite element simulations},
    author = {Patrick J. Blonigan and John Tencer and Sean Babiniec and Jonathan Murray},
    booktitle = {AIAA SciTech 2025 Forum},
    pages = {2133},
    year = {2025},
    doi = {10.2514/6.2025-2133},
}

@article{boef2024stablesparse,
    title = {Stable sparse operator inference for nonlinear structural dynamics},
    author = {Pascal {den Boef} and Diana Manvelyan and Joseph Maubach and Wil Schilders and Nathan {van de Wouw}},
    journal = {arXiv},
    volume = {2407.21672},
    year = {2024},
    doi = {10.48550/arXiv.2407.21672},
}

@article{bychkov2024quadratization,
    title = {Exact and optimal quadratization of nonlinear finite-dimensional non-autonomous dynamical systems},
    author = {Andrey Bychkov and Opal Issan and Gleb Pogudin and Boris Kramer},
    journal = {SIAM Journal of Applied Dynamical Systems},
    volume = {23},
    number = {1},
    pages = {982-1016},
    year = {2024},
    doi = {10.1137/23M1561129},
}

@inproceedings{farcas2023parametric,
    title = {Parametric non-intrusive reduced-order models via operator inference for large-scale rotating detonation engine simulations},
    author = {Ionut-Gabriel Farcas and Rayomand Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    booktitle = {AIAA SciTech 2023 Forum},
    pages = {0172},
    year = {2023},
    doi = {10.2514/6.2023-0172},
}

@article{farcas2024domaindecomposition,
    title = {Domain decomposition for data-driven reduced modeling of large-scale systems},
    author = {Ionut-Gabriel Farcas and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    journal = {AIAA Journal},
    volume = {62},
    number = {11},
    pages = {4071-4086},
    year = {2024},
    doi = {10.2514/1.J063715},
}

@article{farcas2025distributed,
    title = {Distributed computing for physics-based data-driven reduced modeling at scale: {A}pplication to a rotating detonation rocket engine},
    author = {Ionut-Gabriel Farcaş and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    journal = {Computer Physics Communications},
    volume = {313},
    pages = {109619},
    year = {2025},
    doi = {10.1016/j.cpc.2025.109619},
}

@inbook{farcas2025parallel,
    title = {A parallel implementation of reduced-order modeling of large-scale systems},
    author = {Ionut-Gabriel Farcas and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    booktitle = {AIAA SciTech 2025 Forum},
    pages = {1170},
    year = {2025},
    doi = {10.2514/6.2025-1170},
}

@article{filanova2023mechanical,
    title = {An operator inference oriented approach for linear mechanical systems},
    author = {Yevgeniya Filanova and Igor Pontes Duff and Pawan Goyal and Peter Benner},
    journal = {Mechanical Systems and Signal Processing},
    volume = {200},
    pages = {110620},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.ymssp.2023.110620},
}

@misc{freitag2024stochastic,
    title = {Learning stochastic reduced models from data: a nonintrusive approach},
    author = {Melina A. Freitag and Jan Martin Nicolaus and Martin Redmann},
    journal = {arXiv},
    volume = {2407.05724},
    year = {2024},
    doi = {10.48550/arXiv.2407.05724},
}

@article{geelen2022localized,
    title = {Localized non-intrusive reduced-order modelling in the operator inference framework},
    author = {Rudy Geelen and Karen Willcox},
    journal = {Philosophical Transactions of the Royal Society A},
    volume = {380},
    number = {2229},
    pages = {20210206},
    year = {2022},
    doi = {10.1098/rsta.2021.0206},
}

@inproceedings{geelen2023latent,
    author = {Rudy Geelen and Laura Balzano and Karen Willcox},
    booktitle = {62nd IEEE Conference on Decision and Control (CDC)},
    title = {Learning latent representations in high-dimensional state spaces using polynomial manifold constructions},
    year = {2023},
    volume = {},
    number = {},
    pages = {4960-4965},
    doi = {10.1109/CDC49753.2023.10384209},
}

@article{geelen2023quadmanifold,
    title = {Operator inference for non-intrusive model reduction with quadratic manifolds},
    author = {Rudy Geelen and Stephen Wright and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {403},
    pages = {115717},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2022.115717},
}

@article{geelen2024nonlinmanifold,
    title = {Learning physics-based reduced-order models from data using nonlinear manifolds},
    author = {Rudy Geelen and Laura Balzano and Stephen Wright and Karen Willcox},
    journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
    volume = {34},
    number = {3},
    pages = {033122},
    year = {2024},
    doi = {10.1063/5.0170105},
}

@article{geng2024gradient,
    title = {Gradient preserving {O}perator {I}nference: {D}ata-driven reduced-order models for equations with gradient structure},
    author = {Yuwei Geng and Jasdeep Singh and Lili Ju and Boris Kramer and Zhu Wang},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {427},
    pages = {117033},
    year = {2024},
    doi = {10.1016/j.cma.2024.117033},
}

@article{geng2025porthamiltonian,
    title = {Data-driven reduced-order models for port-{H}amiltonian systems with {O}perator {I}nference},
    author = {Yuwei Geng and Lili Ju and Boris Kramer and Zhu Wang},
    journal = {arXiv},
    volume = {2501.02183},
    year = {2025},
    doi = {10.48550/arXiv.2501.02183},
}

@article{ghattas2021acta,
    title = {Learning physics-based models from data: {P}erspectives from inverse problems and model reduction},
    author = {Omar Ghattas and Karen Willcox},
    journal = {Acta Numerica},
    volume = {30},
    pages = {445--554},
    year = {2021},
    publisher = {Cambridge University Press},
    doi = {10.1017/S0962492921000064},
}

@article{gkimisis2025spatiallylocal,
    title = {Non-intrusive reduced-order modeling for dynamical systems with spatially localized features},
    author = {Leonidas Gkimisis and Nicole Aretz and Marco Tezzele and Thomas Richter and Peter Benner and Karen E. Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {444},
    pages = {118115},
    year = {2025},
    doi = {10.1016/j.cma.2025.118115},
}

@article{gruber2023hamiltonian,
    title = {Canonical and noncanonical {H}amiltonian operator inference},
    author = {Anthony Gruber and Irina Tezaur},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {416},
    year = {2023},
    doi = {10.1016/j.cma.2023.116334},
}

@article{gruber2025variational,
    author = {Anthony Gruber and Irina Tezaur},
    title = {Variationally consistent {H}amiltonian model reduction},
    journal = {SIAM Journal on Applied Dynamical Systems},
    volume = {24},
    number = {1},
    pages = {376-414},
    year = {2025},
    doi = {10.1137/24M1652490},
}

@article{guo2022bayesopinf,
    title = {Bayesian operator inference for data-driven reduced-order modeling},
    author = {Mengwu Guo and Shane A McQuarrie and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {402},
    pages = {115336},
    year = {2022},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2022.115336},
}

@article{issan2023shifted,
    title = {Predicting solar wind streams from the inner-heliosphere to Earth via shifted operator inference},
    author = {Opal Issan and Boris Kramer},
    journal = {Journal of Computational Physics},
    volume = {473},
    pages = {111689},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.jcp.2022.111689},
}

@inproceedings{jain2021performance,
    title = {Performance comparison of data-driven reduced models for a single-injector combustion process},
    author = {Parikshit Jain and Shane McQuarrie and Boris Kramer},
    booktitle = {AIAA Propulsion and Energy 2021 Forum},
    pages = {3633},
    year = {2021},
    doi = {10.2514/6.2021-3633},
}

@article{kang2025semiconductor,
    title = {Parametric {O}perator {I}nference to simulate the purging process in semiconductor manufacturing},
    author = {Seunghyon Kang and Hyeonghun Kim and Boris Kramer},
    journal = {arXiv},
    volume = {2504.03990},
    year = {2025},
    doi = {10.48550/arXiv.2504.03990},
}

@article{khodabakhshi2022diffalg,
    title = {Non-intrusive data-driven model reduction for differential algebraic equations derived from lifting transformations},
    author = {Parisa Khodabakhshi and Karen E. Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {389},
    pages = {114296},
    year = {2022},
    doi = {10.1016/j.cma.2021.114296},
}

@article{kim2025stateconstraints,
    title = {Physically consistent predictive reduced-order modeling by enhancing {O}perator {I}nference with state constraints},
    author = {Hyeonghun Kim and Boris Kramer},
    journal = {arXiv},
    volume = {2502.03672},
    year = {2025},
    doi = {10.48550/arXiv.2502.03672},
}

@inproceedings{koike2024energy,
    title = {Energy-preserving reduced operator inference for efficient design and control},
    author = {Tomoki Koike and Elizabeth Qian},
    booktitle = {AIAA SciTech 2024 Forum},
    pages = {1012},
    year = {2024},
    doi = {10.2514/6.2024-1012},
}

@article{kramer2021quadstability,
    title = {Stability domains for quadratic-bilinear reduced-order models},
    author = {Boris Kramer},
    journal = {SIAM Journal on Applied Dynamical Systems},
    volume = {20},
    issue = {2},
    pages = {981--996},
    year = {2021},
    publisher = {SIAM},
    doi = {10.1137/20M1364849},
}

@article{kramer2024survey,
    title = {Learning nonlinear reduced models from data with operator inference},
    author = {Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
    journal = {Annual Review of Fluid Mechanics},
    volume = {56},
    pages = {521--548},
    year = {2024},
    publisher = {Annual Reviews},
    doi = {10.1146/annurev-fluid-121021-025220},
}

@article{mcquarrie2021combustion,
    title = {Data-driven reduced-order models via regularised operator inference for a single-injector combustion process},
    author = {Shane A McQuarrie and Cheng Huang and Karen Willcox},
    journal = {Journal of the Royal Society of New Zealand},
    volume = {51},
    issue = {2},
    pages = {194-211},
    year = {2021},
    publisher = {Taylor \& Francis},
    doi = {10.1080/03036758.2020.1863237},
}

@article{mcquarrie2023parametric,
    title = {Nonintrusive reduced-order models for parametric partial differential equations via data-driven operator inference},
    author = {Shane A McQuarrie and Parisa Khodabakhshi and Karen Willcox},
    journal = {SIAM Journal on Scientific Computing},
    volume = {45},
    issue = {4},
    pages = {A1917-A1946},
    year = {2023},
    publisher = {SIAM},
    doi = {10.1137/21M1452810},
}

@phdthesis{mcquarrie2023thesis,
    title = {Data-driven parametric reduced-order models: {O}perator inference for reactive flow applications},
    author = {Shane Alexander McQuarrie},
    school = {The University of Texas at Austin},
    year = {2023},
    doi = {10.26153/tsw/50172},
}

@article{mcquarrie2025gpbayesopinf,
    title = {Bayesian learning with {G}aussian processes for low-dimensional representations of time-dependent nonlinear systems},
    author = {Shane A. McQuarrie and Anirban Chaudhuri and Karen E. Willcox and Mengwu Guo},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {475},
    pages = {134572},
    year = {2025},
    doi = {10.1016/j.physd.2025.134572},
}

@article{peherstorfer2016opinf,
    title = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author = {Benjamin Peherstorfer and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {306},
    pages = {196--215},
    year = {2016},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2016.03.025},
}

@article{peherstorfer2020reprojection,
    title = {Sampling low-dimensional {M}arkovian dynamics for preasymptotically recovering reduced models from data with operator inference},
    author = {Benjamin Peherstorfer},
    journal = {SIAM Journal on Scientific Computing},
    volume = {42},
    issue = {5},
    pages = {A3489-a3515},
    year = {2020},
    publisher = {SIAM},
    doi = {10.1137/19M1292448},
}

@inproceedings{qian2019transform,
    title = {Transform \& {L}earn: {A} data-driven approach to nonlinear model reduction},
    author = {Elizabeth Qian and Boris Kramer and Alexandre N. Marques and Karen E. Willcox},
    booktitle = {AIAA Aviation 2019 Forum},
    pages = {3707},
    year = {2019},
    doi = {10.2514/6.2019-3707},
}

@article{qian2020liftandlearn,
    title = {Lift \& {L}earn: {P}hysics-informed machine learning for large-scale nonlinear dynamical systems},
    author = {Elizabeth Qian and Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {406},
    pages = {132401},
    year = {2020},
    publisher = {Elsevier},
    doi = {10.1016/j.physd.2020.132401},
}

@phdthesis{qian2021thesis,
    title = {A scientific machine learning approach to learning reduced models for nonlinear partial differential equations},
    author = {Elizabeth Yi Qian},
    school = {Massachusetts Institute of Technology},
    year = {2021},
}

@article{qian2022pdes,
    title = {Reduced operator inference for nonlinear partial differential equations},
    author = {Elizabeth Qian and Ionut-Gabriel Farcas and Karen Willcox},
    journal = {SIAM Journal on Scientific Computing},
    volume = {44},
    issue = {4},
    pages = {A1934-a1959},
    year = {2022},
    publisher = {SIAM},
    doi = {10.1137/21M1393972},
}

@inproceedings{rocha2022c02,
    title = {Data-driven reduced-order model for atmospheric {CO}2 dispersion},
    author = {Pedro Roberto Barbosa Rocha and Marcos Sebasti\~{a}o de Paula Gomes and Jo\~{a}o Lucas de Sousa Almeida and Allan M Carvalho and Alberto Costa Nogueira Jr},
    booktitle = {AAAI Fall Symposium},
    year = {2022},
}

@article{rocha2023convection,
    title = {Reduced-order modeling of the two-dimensional {R}ayleigh--{B}\'{e}nard convection flow through a non-intrusive operator inference},
    author = {Pedro Roberto Barbosa Rocha and Jo\~{a}o Lucas de Sousa Almeida and Marcos Sebasti\~{a}o de Paula Gomes and Alberto Costa Nogueira Jr},
    journal = {Engineering Applications of Artificial Intelligence},
    volume = {126},
    pages = {106923},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.engappai.2023.106923},
}

@article{rosenberger2025exactopinf,
    title = {Exact operator inference with minimal data},
    author = {Henrik Rosenberger and Benjamin Sanderse and Giovanni Stabile},
    journal = {arXiv},
    volume = {2506.01244},
    year = {2025},
    doi = {10.48550/arXiv.2506.01244},
}

@mastersthesis{salinger2021thesis,
    title = {Toward predictive digital twins for self-aware unmanned aerial vehicles: {N}on-intrusive reduced order models and experimental data analysis},
    author = {Stephanie Joyce Salinger},
    school = {The University of Texas at Austin},
    year = {2021},
}

@article{sawant2023pireg,
    title = {Physics-informed regularization and structure preservation for learning stable reduced models from data with operator inference},
    author = {Nihar Sawant and Boris Kramer and Benjamin Peherstorfer},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {404},
    pages = {115836},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2022.115836},
}

@phdthesis{sawant2023thesis,
    title = {Learning structured and stable reduced models from data with operator inference},
    author = {Nihar Sawant},
    year = {2023},
    school = {New York University},
}

@article{sharma2022hamiltonian,
    title = {Hamiltonian operator inference: {P}hysics-preserving learning of reduced-order models for canonical {H}amiltonian systems},
    author = {Harsh Sharma and Zhu Wang and Boris Kramer},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {431},
    pages = {133122},
    year = {2022},
    publisher = {Elsevier},
    doi = {10.1016/j.physd.2021.133122},
}

@article{sharma2024lagrangian,
    title = {Lagrangian operator inference enhanced with structure-preserving machine learning for nonintrusive model reduction of mechanical systems},
    author = {Harsh Sharma and David A Najera-Flores and Michael D Todd and Boris Kramer},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {423},
    pages = {116865},
    year = {2024},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2024.116865},
}

@article{sharma2024preserving,
    title = {Preserving {L}agrangian structure in data-driven reduced-order modeling of large-scale mechanical systems},
    author = {Harsh Sharma and Boris Kramer},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {462},
    pages = {134128},
    year = {2024},
    doi = {10.1016/j.physd.2024.134128},
}

@mastersthesis{swischuk2019thesis,
    title = {Physics-based machine learning and data-driven reduced-order modeling},
    author = {Renee Copland Swischuk},
    school = {Massachusetts Institute of Technology},
    year = {2019},
}

@article{swischuk2020combustion,
    title = {Learning physics-based reduced-order models for a single-injector combustion process},
    author = {Renee Swischuk and Boris Kramer and Cheng Huang and Karen Willcox},
    journal = {AIAA Journal},
    volume = {58},
    issue = {6},
    pages = {2658--2672},
    year = {2020},
    publisher = {American Institute of Aeronautics and Astronautics},
    doi = {10.2514/1.J058943},
}

@article{uy2021error,
    title = {Probabilistic error estimation for non-intrusive reduced models learned from data of systems governed by linear parabolic partial differential equations},
    author = {Wayne Isaac Tan Uy and Benjamin Peherstorfer},
    journal = {ESAIM: Mathematical Modelling and Numerical Analysis},
    volume = {55},
    issue = {3},
    pages = {735--761},
    year = {2021},
    publisher = {EDP Sciences},
    doi = {10.1051/m2an/2021010},
}

@article{uy2021partial,
    title = {Operator inference of non-{M}arkovian terms for learning reduced models from partially observed state trajectories},
    author = {Wayne Isaac Tan Uy and Benjamin Peherstorfer},
    journal = {Journal of Scientific Computing},
    volume = {88},
    issue = {3},
    pages = {1--31},
    year = {2021},
    publisher = {Springer},
    doi = {10.1007/s10915-021-01580-2},
}

@article{uy2023active,
    title = {Active operator inference for learning low-dimensional dynamical-system models from noisy data},
    author = {Wayne Isaac Tan Uy and Yuepeng Wang and Yuxiao Wen and Benjamin Peherstorfer},
    journal = {SIAM Journal on Scientific Computing},
    volume = {45},
    issue = {4},
    pages = {A1462-a1490},
    year = {2023},
    publisher = {SIAM},
    doi = {10.1137/21M1439729},
}

@article{uy2023rollouts,
    title = {Operator inference with roll outs for learning reduced models from scarce and low-quality data},
    author = {Wayne Isaac Tan Uy and Dirk Hartmann and Benjamin Peherstorfer},
    journal = {Computers \& Mathematics with Applications},
    volume = {145},
    pages = {224--239},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.camwa.2023.06.012},
}

@article{vijaywargiya2025tensoropinf,
    title = {Tensor parametric {H}amiltonian operator inference},
    author = {Arjun Vijaywargiya and Shane A. McQuarrie and Anthony Gruber},
    journal = {arXiv},
    volume = {2502.10888},
    year = {2025},
    doi = {10.48550/arXiv.2502.10888},
}

@article{yildiz2021shallow,
    title = {Learning reduced-order dynamics for parametrized shallow water equations from data},
    author = {S\"{u}leyman Y\i{}ld\i{}z and Pawan Goyal and Peter Benner and B\"{u}lent Karas\"{o}zen},
    journal = {International Journal for Numerical Methods in Fluids},
    volume = {93},
    issue = {8},
    pages = {2803--2821},
    year = {2021},
    publisher = {Wiley Online Library},
    doi = {10.1002/fld.4998},
}

@inproceedings{zastrow2023flutter,
    title = {Data-driven model reduction via operator inference for coupled aeroelastic flutter},
    author = {Benjamin G Zastrow and Anirban Chaudhuri and Karen Willcox and Anthony S Ashley and Michael C Henson},
    booktitle = {AIAA SciTech 2023 Forum},
    pages = {0330},
    year = {2023},
    doi = {10.2514/6.2023-0330},
}

```
:::

:::{admonition} Sorted by year then alphabetically by author
:class: dropdown seealso

```bibtex
@article{peherstorfer2016opinf,
    title = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author = {Benjamin Peherstorfer and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {306},
    pages = {196--215},
    year = {2016},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2016.03.025},
}

@inproceedings{qian2019transform,
    title = {Transform \& {L}earn: {A} data-driven approach to nonlinear model reduction},
    author = {Elizabeth Qian and Boris Kramer and Alexandre N. Marques and Karen E. Willcox},
    booktitle = {AIAA Aviation 2019 Forum},
    pages = {3707},
    year = {2019},
    doi = {10.2514/6.2019-3707},
}

@mastersthesis{swischuk2019thesis,
    title = {Physics-based machine learning and data-driven reduced-order modeling},
    author = {Renee Copland Swischuk},
    school = {Massachusetts Institute of Technology},
    year = {2019},
}

@article{benner2020deim,
    title = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
    author = {Peter Benner and Pawan Goyal and Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {372},
    pages = {113433},
    year = {2020},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2020.113433},
}

@article{peherstorfer2020reprojection,
    title = {Sampling low-dimensional {M}arkovian dynamics for preasymptotically recovering reduced models from data with operator inference},
    author = {Benjamin Peherstorfer},
    journal = {SIAM Journal on Scientific Computing},
    volume = {42},
    issue = {5},
    pages = {A3489-a3515},
    year = {2020},
    publisher = {SIAM},
    doi = {10.1137/19M1292448},
}

@article{qian2020liftandlearn,
    title = {Lift \& {L}earn: {P}hysics-informed machine learning for large-scale nonlinear dynamical systems},
    author = {Elizabeth Qian and Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {406},
    pages = {132401},
    year = {2020},
    publisher = {Elsevier},
    doi = {10.1016/j.physd.2020.132401},
}

@article{swischuk2020combustion,
    title = {Learning physics-based reduced-order models for a single-injector combustion process},
    author = {Renee Swischuk and Boris Kramer and Cheng Huang and Karen Willcox},
    journal = {AIAA Journal},
    volume = {58},
    issue = {6},
    pages = {2658--2672},
    year = {2020},
    publisher = {American Institute of Aeronautics and Astronautics},
    doi = {10.2514/1.J058943},
}

@article{ghattas2021acta,
    title = {Learning physics-based models from data: {P}erspectives from inverse problems and model reduction},
    author = {Omar Ghattas and Karen Willcox},
    journal = {Acta Numerica},
    volume = {30},
    pages = {445--554},
    year = {2021},
    publisher = {Cambridge University Press},
    doi = {10.1017/S0962492921000064},
}

@inproceedings{jain2021performance,
    title = {Performance comparison of data-driven reduced models for a single-injector combustion process},
    author = {Parikshit Jain and Shane McQuarrie and Boris Kramer},
    booktitle = {AIAA Propulsion and Energy 2021 Forum},
    pages = {3633},
    year = {2021},
    doi = {10.2514/6.2021-3633},
}

@article{kramer2021quadstability,
    title = {Stability domains for quadratic-bilinear reduced-order models},
    author = {Boris Kramer},
    journal = {SIAM Journal on Applied Dynamical Systems},
    volume = {20},
    issue = {2},
    pages = {981--996},
    year = {2021},
    publisher = {SIAM},
    doi = {10.1137/20M1364849},
}

@article{mcquarrie2021combustion,
    title = {Data-driven reduced-order models via regularised operator inference for a single-injector combustion process},
    author = {Shane A McQuarrie and Cheng Huang and Karen Willcox},
    journal = {Journal of the Royal Society of New Zealand},
    volume = {51},
    issue = {2},
    pages = {194-211},
    year = {2021},
    publisher = {Taylor \& Francis},
    doi = {10.1080/03036758.2020.1863237},
}

@phdthesis{qian2021thesis,
    title = {A scientific machine learning approach to learning reduced models for nonlinear partial differential equations},
    author = {Elizabeth Yi Qian},
    school = {Massachusetts Institute of Technology},
    year = {2021},
}

@mastersthesis{salinger2021thesis,
    title = {Toward predictive digital twins for self-aware unmanned aerial vehicles: {N}on-intrusive reduced order models and experimental data analysis},
    author = {Stephanie Joyce Salinger},
    school = {The University of Texas at Austin},
    year = {2021},
}

@article{uy2021partial,
    title = {Operator inference of non-{M}arkovian terms for learning reduced models from partially observed state trajectories},
    author = {Wayne Isaac Tan Uy and Benjamin Peherstorfer},
    journal = {Journal of Scientific Computing},
    volume = {88},
    issue = {3},
    pages = {1--31},
    year = {2021},
    publisher = {Springer},
    doi = {10.1007/s10915-021-01580-2},
}

@article{uy2021error,
    title = {Probabilistic error estimation for non-intrusive reduced models learned from data of systems governed by linear parabolic partial differential equations},
    author = {Wayne Isaac Tan Uy and Benjamin Peherstorfer},
    journal = {ESAIM: Mathematical Modelling and Numerical Analysis},
    volume = {55},
    issue = {3},
    pages = {735--761},
    year = {2021},
    publisher = {EDP Sciences},
    doi = {10.1051/m2an/2021010},
}

@article{yildiz2021shallow,
    title = {Learning reduced-order dynamics for parametrized shallow water equations from data},
    author = {S\"{u}leyman Y\i{}ld\i{}z and Pawan Goyal and Peter Benner and B\"{u}lent Karas\"{o}zen},
    journal = {International Journal for Numerical Methods in Fluids},
    volume = {93},
    issue = {8},
    pages = {2803--2821},
    year = {2021},
    publisher = {Wiley Online Library},
    doi = {10.1002/fld.4998},
}

@article{almeida2022chaotic,
    title = {Non-Intrusive reduced models based on operator inference for chaotic systems},
    author = {Jo\~{a}o Lucas de Sousa Almeida and Arthur Cancellieri Pires and Klaus Feine Vaz Cid and Alberto Costa Nogueira Jr},
    journal = {arXiv},
    volume = {2206.01604},
    year = {2022},
}

@article{benner2022incompressible,
    title = {Operator inference and physics-informed learning of low-dimensional models for incompressible flows},
    author = {Peter Benner and Pawan Goyal and Jan Heiland and Igor Pontes Duff},
    journal = {Electronic Transactions on Numerical Analysis},
    volume = {56},
    year = {2022},
    doi = {10.1553/etna_vol56s28},
}

@article{geelen2022localized,
    title = {Localized non-intrusive reduced-order modelling in the operator inference framework},
    author = {Rudy Geelen and Karen Willcox},
    journal = {Philosophical Transactions of the Royal Society A},
    volume = {380},
    number = {2229},
    pages = {20210206},
    year = {2022},
    doi = {10.1098/rsta.2021.0206},
}

@article{guo2022bayesopinf,
    title = {Bayesian operator inference for data-driven reduced-order modeling},
    author = {Mengwu Guo and Shane A McQuarrie and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {402},
    pages = {115336},
    year = {2022},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2022.115336},
}

@article{khodabakhshi2022diffalg,
    title = {Non-intrusive data-driven model reduction for differential algebraic equations derived from lifting transformations},
    author = {Parisa Khodabakhshi and Karen E. Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {389},
    pages = {114296},
    year = {2022},
    doi = {10.1016/j.cma.2021.114296},
}

@article{qian2022pdes,
    title = {Reduced operator inference for nonlinear partial differential equations},
    author = {Elizabeth Qian and Ionut-Gabriel Farcas and Karen Willcox},
    journal = {SIAM Journal on Scientific Computing},
    volume = {44},
    issue = {4},
    pages = {A1934-a1959},
    year = {2022},
    publisher = {SIAM},
    doi = {10.1137/21M1393972},
}

@inproceedings{rocha2022c02,
    title = {Data-driven reduced-order model for atmospheric {CO}2 dispersion},
    author = {Pedro Roberto Barbosa Rocha and Marcos Sebasti\~{a}o de Paula Gomes and Jo\~{a}o Lucas de Sousa Almeida and Allan M Carvalho and Alberto Costa Nogueira Jr},
    booktitle = {AAAI Fall Symposium},
    year = {2022},
}

@article{sharma2022hamiltonian,
    title = {Hamiltonian operator inference: {P}hysics-preserving learning of reduced-order models for canonical {H}amiltonian systems},
    author = {Harsh Sharma and Zhu Wang and Boris Kramer},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {431},
    pages = {133122},
    year = {2022},
    publisher = {Elsevier},
    doi = {10.1016/j.physd.2021.133122},
}

@article{benner2023quaddecoder,
    title = {A quadratic decoder approach to nonintrusive reduced-order modeling of nonlinear dynamical systems},
    author = {Peter Benner and Pawan Goyal and Jan Heiland and Igor Pontes Duff},
    journal = {Proceedings in Applied Mathematics and Mechanics},
    volume = {23},
    number = {1},
    pages = {e202200049},
    year = {2023},
    doi = {10.1002/pamm.202200049},
}

@inproceedings{farcas2023parametric,
    title = {Parametric non-intrusive reduced-order models via operator inference for large-scale rotating detonation engine simulations},
    author = {Ionut-Gabriel Farcas and Rayomand Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    booktitle = {AIAA SciTech 2023 Forum},
    pages = {0172},
    year = {2023},
    doi = {10.2514/6.2023-0172},
}

@article{filanova2023mechanical,
    title = {An operator inference oriented approach for linear mechanical systems},
    author = {Yevgeniya Filanova and Igor Pontes Duff and Pawan Goyal and Peter Benner},
    journal = {Mechanical Systems and Signal Processing},
    volume = {200},
    pages = {110620},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.ymssp.2023.110620},
}

@article{geelen2023quadmanifold,
    title = {Operator inference for non-intrusive model reduction with quadratic manifolds},
    author = {Rudy Geelen and Stephen Wright and Karen Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {403},
    pages = {115717},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2022.115717},
}

@inproceedings{geelen2023latent,
    author = {Rudy Geelen and Laura Balzano and Karen Willcox},
    booktitle = {62nd IEEE Conference on Decision and Control (CDC)},
    title = {Learning latent representations in high-dimensional state spaces using polynomial manifold constructions},
    year = {2023},
    volume = {},
    number = {},
    pages = {4960-4965},
    doi = {10.1109/CDC49753.2023.10384209},
}

@article{gruber2023hamiltonian,
    title = {Canonical and noncanonical {H}amiltonian operator inference},
    author = {Anthony Gruber and Irina Tezaur},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {416},
    year = {2023},
    doi = {10.1016/j.cma.2023.116334},
}

@article{issan2023shifted,
    title = {Predicting solar wind streams from the inner-heliosphere to Earth via shifted operator inference},
    author = {Opal Issan and Boris Kramer},
    journal = {Journal of Computational Physics},
    volume = {473},
    pages = {111689},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.jcp.2022.111689},
}

@article{mcquarrie2023parametric,
    title = {Nonintrusive reduced-order models for parametric partial differential equations via data-driven operator inference},
    author = {Shane A McQuarrie and Parisa Khodabakhshi and Karen Willcox},
    journal = {SIAM Journal on Scientific Computing},
    volume = {45},
    issue = {4},
    pages = {A1917-A1946},
    year = {2023},
    publisher = {SIAM},
    doi = {10.1137/21M1452810},
}

@phdthesis{mcquarrie2023thesis,
    title = {Data-driven parametric reduced-order models: {O}perator inference for reactive flow applications},
    author = {Shane Alexander McQuarrie},
    school = {The University of Texas at Austin},
    year = {2023},
    doi = {10.26153/tsw/50172},
}

@article{rocha2023convection,
    title = {Reduced-order modeling of the two-dimensional {R}ayleigh--{B}\'{e}nard convection flow through a non-intrusive operator inference},
    author = {Pedro Roberto Barbosa Rocha and Jo\~{a}o Lucas de Sousa Almeida and Marcos Sebasti\~{a}o de Paula Gomes and Alberto Costa Nogueira Jr},
    journal = {Engineering Applications of Artificial Intelligence},
    volume = {126},
    pages = {106923},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.engappai.2023.106923},
}

@article{sawant2023pireg,
    title = {Physics-informed regularization and structure preservation for learning stable reduced models from data with operator inference},
    author = {Nihar Sawant and Boris Kramer and Benjamin Peherstorfer},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {404},
    pages = {115836},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2022.115836},
}

@phdthesis{sawant2023thesis,
    title = {Learning structured and stable reduced models from data with operator inference},
    author = {Nihar Sawant},
    year = {2023},
    school = {New York University},
}

@article{uy2023active,
    title = {Active operator inference for learning low-dimensional dynamical-system models from noisy data},
    author = {Wayne Isaac Tan Uy and Yuepeng Wang and Yuxiao Wen and Benjamin Peherstorfer},
    journal = {SIAM Journal on Scientific Computing},
    volume = {45},
    issue = {4},
    pages = {A1462-a1490},
    year = {2023},
    publisher = {SIAM},
    doi = {10.1137/21M1439729},
}

@article{uy2023rollouts,
    title = {Operator inference with roll outs for learning reduced models from scarce and low-quality data},
    author = {Wayne Isaac Tan Uy and Dirk Hartmann and Benjamin Peherstorfer},
    journal = {Computers \& Mathematics with Applications},
    volume = {145},
    pages = {224--239},
    year = {2023},
    publisher = {Elsevier},
    doi = {10.1016/j.camwa.2023.06.012},
}

@inproceedings{zastrow2023flutter,
    title = {Data-driven model reduction via operator inference for coupled aeroelastic flutter},
    author = {Benjamin G Zastrow and Anirban Chaudhuri and Karen Willcox and Anthony S Ashley and Michael C Henson},
    booktitle = {AIAA SciTech 2023 Forum},
    pages = {0330},
    year = {2023},
    doi = {10.2514/6.2023-0330},
}

@inproceedings{aretz2024enforcing,
    title = {Enforcing structure in data-driven reduced modeling through nested {O}perator {I}nference},
    author = {Nicole Aretz and Karen Willcox},
    booktitle = {63rd IEEE Conference on Decision and Control (CDC)},
    year = {2024},
    organization = {IEEE},
    doi = {10.1109/CDC56724.2024.10885857},
}

@article{boef2024stablesparse,
    title = {Stable sparse operator inference for nonlinear structural dynamics},
    author = {Pascal {den Boef} and Diana Manvelyan and Joseph Maubach and Wil Schilders and Nathan {van de Wouw}},
    journal = {arXiv},
    volume = {2407.21672},
    year = {2024},
    doi = {10.48550/arXiv.2407.21672},
}

@article{bychkov2024quadratization,
    title = {Exact and optimal quadratization of nonlinear finite-dimensional non-autonomous dynamical systems},
    author = {Andrey Bychkov and Opal Issan and Gleb Pogudin and Boris Kramer},
    journal = {SIAM Journal of Applied Dynamical Systems},
    volume = {23},
    number = {1},
    pages = {982-1016},
    year = {2024},
    doi = {10.1137/23M1561129},
}

@article{farcas2024domaindecomposition,
    title = {Domain decomposition for data-driven reduced modeling of large-scale systems},
    author = {Ionut-Gabriel Farcas and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    journal = {AIAA Journal},
    volume = {62},
    number = {11},
    pages = {4071-4086},
    year = {2024},
    doi = {10.2514/1.J063715},
}

@misc{freitag2024stochastic,
    title = {Learning stochastic reduced models from data: a nonintrusive approach},
    author = {Melina A. Freitag and Jan Martin Nicolaus and Martin Redmann},
    journal = {arXiv},
    volume = {2407.05724},
    year = {2024},
    doi = {10.48550/arXiv.2407.05724},
}

@article{geelen2024nonlinmanifold,
    title = {Learning physics-based reduced-order models from data using nonlinear manifolds},
    author = {Rudy Geelen and Laura Balzano and Stephen Wright and Karen Willcox},
    journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
    volume = {34},
    number = {3},
    pages = {033122},
    year = {2024},
    doi = {10.1063/5.0170105},
}

@article{geng2024gradient,
    title = {Gradient preserving {O}perator {I}nference: {D}ata-driven reduced-order models for equations with gradient structure},
    author = {Yuwei Geng and Jasdeep Singh and Lili Ju and Boris Kramer and Zhu Wang},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {427},
    pages = {117033},
    year = {2024},
    doi = {10.1016/j.cma.2024.117033},
}

@inproceedings{koike2024energy,
    title = {Energy-preserving reduced operator inference for efficient design and control},
    author = {Tomoki Koike and Elizabeth Qian},
    booktitle = {AIAA SciTech 2024 Forum},
    pages = {1012},
    year = {2024},
    doi = {10.2514/6.2024-1012},
}

@article{kramer2024survey,
    title = {Learning nonlinear reduced models from data with operator inference},
    author = {Boris Kramer and Benjamin Peherstorfer and Karen Willcox},
    journal = {Annual Review of Fluid Mechanics},
    volume = {56},
    pages = {521--548},
    year = {2024},
    publisher = {Annual Reviews},
    doi = {10.1146/annurev-fluid-121021-025220},
}

@article{sharma2024lagrangian,
    title = {Lagrangian operator inference enhanced with structure-preserving machine learning for nonintrusive model reduction of mechanical systems},
    author = {Harsh Sharma and David A Najera-Flores and Michael D Todd and Boris Kramer},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {423},
    pages = {116865},
    year = {2024},
    publisher = {Elsevier},
    doi = {10.1016/j.cma.2024.116865},
}

@article{sharma2024preserving,
    title = {Preserving {L}agrangian structure in data-driven reduced-order modeling of large-scale mechanical systems},
    author = {Harsh Sharma and Boris Kramer},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {462},
    pages = {134128},
    year = {2024},
    doi = {10.1016/j.physd.2024.134128},
}

@inbook{blonigan2025thermal,
    title = {Operator inference-based model order reduction of thermal protection system finite element simulations},
    author = {Patrick J. Blonigan and John Tencer and Sean Babiniec and Jonathan Murray},
    booktitle = {AIAA SciTech 2025 Forum},
    pages = {2133},
    year = {2025},
    doi = {10.2514/6.2025-2133},
}

@inbook{farcas2025parallel,
    title = {A parallel implementation of reduced-order modeling of large-scale systems},
    author = {Ionut-Gabriel Farcas and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    booktitle = {AIAA SciTech 2025 Forum},
    pages = {1170},
    year = {2025},
    doi = {10.2514/6.2025-1170},
}

@article{farcas2025distributed,
    title = {Distributed computing for physics-based data-driven reduced modeling at scale: {A}pplication to a rotating detonation rocket engine},
    author = {Ionut-Gabriel Farcaş and Rayomand P. Gundevia and Ramakanth Munipalli and Karen E. Willcox},
    journal = {Computer Physics Communications},
    volume = {313},
    pages = {109619},
    year = {2025},
    doi = {10.1016/j.cpc.2025.109619},
}

@article{geng2025porthamiltonian,
    title = {Data-driven reduced-order models for port-{H}amiltonian systems with {O}perator {I}nference},
    author = {Yuwei Geng and Lili Ju and Boris Kramer and Zhu Wang},
    journal = {arXiv},
    volume = {2501.02183},
    year = {2025},
    doi = {10.48550/arXiv.2501.02183},
}

@article{gkimisis2025spatiallylocal,
    title = {Non-intrusive reduced-order modeling for dynamical systems with spatially localized features},
    author = {Leonidas Gkimisis and Nicole Aretz and Marco Tezzele and Thomas Richter and Peter Benner and Karen E. Willcox},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {444},
    pages = {118115},
    year = {2025},
    doi = {10.1016/j.cma.2025.118115},
}

@article{gruber2025variational,
    author = {Anthony Gruber and Irina Tezaur},
    title = {Variationally consistent {H}amiltonian model reduction},
    journal = {SIAM Journal on Applied Dynamical Systems},
    volume = {24},
    number = {1},
    pages = {376-414},
    year = {2025},
    doi = {10.1137/24M1652490},
}

@article{kang2025semiconductor,
    title = {Parametric {O}perator {I}nference to simulate the purging process in semiconductor manufacturing},
    author = {Seunghyon Kang and Hyeonghun Kim and Boris Kramer},
    journal = {arXiv},
    volume = {2504.03990},
    year = {2025},
    doi = {10.48550/arXiv.2504.03990},
}

@article{kim2025stateconstraints,
    title = {Physically consistent predictive reduced-order modeling by enhancing {O}perator {I}nference with state constraints},
    author = {Hyeonghun Kim and Boris Kramer},
    journal = {arXiv},
    volume = {2502.03672},
    year = {2025},
    doi = {10.48550/arXiv.2502.03672},
}

@article{mcquarrie2025gpbayesopinf,
    title = {Bayesian learning with {G}aussian processes for low-dimensional representations of time-dependent nonlinear systems},
    author = {Shane A. McQuarrie and Anirban Chaudhuri and Karen E. Willcox and Mengwu Guo},
    journal = {Physica D: Nonlinear Phenomena},
    volume = {475},
    pages = {134572},
    year = {2025},
    doi = {10.1016/j.physd.2025.134572},
}

@article{rosenberger2025exactopinf,
    title = {Exact operator inference with minimal data},
    author = {Henrik Rosenberger and Benjamin Sanderse and Giovanni Stabile},
    journal = {arXiv},
    volume = {2506.01244},
    year = {2025},
    doi = {10.48550/arXiv.2506.01244},
}

@article{vijaywargiya2025tensoropinf,
    title = {Tensor parametric {H}amiltonian operator inference},
    author = {Arjun Vijaywargiya and Shane A. McQuarrie and Anthony Gruber},
    journal = {arXiv},
    volume = {2502.10888},
    year = {2025},
    doi = {10.48550/arXiv.2502.10888},
}

```
:::
