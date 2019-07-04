
Operator inference package
==========================
**Author: Renee Swischuk (swischuk@mit.edu)**



The first step is to download the operator inference module from pip. In
the command prompt, type <br/>
`pip3 install -i https://test.pypi.org/simple/ operator-inference`<br/>
*This is temporary*<br/><br/>
The `operator_inference` package
contains a `model` class, with
functions defined in
[1.1.1](#sec:modelclassfunctions), and two helper scripts called
`opinf_helper.py` and
`integration_helpers.py`, with
functions defined in
[1.2.1](#sec:opinfhelperfunctions) and
[1.3.1](#sec:integrationhelpersfunctions), respectively.<br/>

 Quick Start
-----------
`from operator_inference import OpInf`<br/>

#define a model of the form <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\hat{\mathbf{x}}}&space;=&space;\mathbf{A}\hat{\mathbf{x}}&space;&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\hat{\mathbf{x}}&space;&plus;&space;\mathbf{c}" title="\dot{\hat{\mathbf{x}}} = \mathbf{A}\hat{\mathbf{x}} + \mathbf{c}" /></a><br/>
`mymodel = OpInf.model('Lc',False) # a linear quadratic with no input` <br/>

#(fit the model) -- solve for the operators that minimize <a href="https://www.codecogs.com/eqnedit.php?latex=\displaystyle&space;\min_{\hat{\mathbf{A}},\hat{\mathbf{c}}}&space;\dot{\hat{\mathbf{x}}}&space;-&space;\mathbf{A}\hat{\mathbf{x}}&space;-&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\displaystyle&space;\min_{\hat{\mathbf{A}},\hat{\mathbf{c}}}&space;\dot{\mathbf{x}}&space;-&space;\mathbf{A}\hat{\mathbf{x}}&space;-&space;\mathbf{c}" title="\displaystyle \min_{\hat{\mathbf{A}},\hat{\mathbf{c}}} \dot{\mathbf{x}} - \mathbf{A}\hat{\mathbf{x}} - \mathbf{c}" /></a> <br/>
`mymodel.fit(r,k,xdot,xhat)`

#simulate the learned model,<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\hat{\mathbf{x}}}&space;=&space;\hat{\mathbf{A}}\hat{\mathbf{x}}&space;&plus;&space;\hat{\mathbf{c}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\hat{\mathbf{x}}}&space;=&space;\hat{\mathbf{A}}\hat{\mathbf{x}}&space;&plus;&space;\hat{\mathbf{c}}" title="\dot{\hat{\mathbf{x}}} = \hat{\mathbf{A}}\hat{\mathbf{x}} + \hat{\mathbf{c}}" /></a> , for n_t time steps<br/>
`xr,break_point = mymodel.predict(xhat[:,0], n_t, dt)`<br/>

#reconstruct the predictions<br/>
`xr_rec = U[:,:r]@xr` <br/>

See `opinf_demo.py` for a working example

 Model class
-----------

The following commands will initialize an operator inference model.<br/>
 >`from operator_inference import operator_inference`<br/>
    `my_model = operator_inference.model(degree, input)`<br/>
    
where `degree` is a string denoting the degree of
the model with the following options

-   'L' -- a linear model, <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}" /></a>

-   'Lc' -- a linear model with a constant, <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{c}" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}+ \mathbf{c}" /></a>
    

-   'LQ' -- a linear and quadratic model, <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{F}\mathbf{x}^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{F}\mathbf{x}^2" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}+ \mathbf{F}\mathbf{x}^2" /></a>

-   'LQc' -- a linear and quadratic model with a constant, <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{F}\mathbf{x}^2&space;&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{F}\mathbf{x}^2&space;&plus;&space;\mathbf{c}" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}+ \mathbf{F}\mathbf{x}^2 + \mathbf{c}" /></a>

-   'Q' -- a quadratic model, <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{F}\mathbf{x}^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{F}\mathbf{x}^2" title="\dot{\mathbf{x}} = \mathbf{F}\mathbf{x}^2" /></a>

-   'Qc' -- a quadratic model with a constant, <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{F}\mathbf{x}^2&space;&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{F}\mathbf{x}^2&space;&plus;&space;\mathbf{c}" title="\dot{\mathbf{x}} = \mathbf{F}\mathbf{x}^2 + \mathbf{c}" /></a>

The `input` argument is a
boolean (True or False) denoting whether or not there is an additive
input term of the form <a href="https://www.codecogs.com/eqnedit.php?latex=&plus;\mathbf{B}&space;\mathbf{U}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?&plus;\mathbf{B}&space;\mathbf{U}" title="+\mathbf{B} \mathbf{U}" /></a>.<br/>
The script, `opinf_demo.py`
demonstrates the use of the operator inference model on data generated
from the heat equation. See [@mythesis] for the problem setup.

### Model class functions 

Functions can be called as
`mymodel.function_name()`

1.  `fit(r,reg,xdot,xhat,u=None)`

    Find the operators of the reduced-order model that fit the data by solving the regularized least
    squares problem <br/>
    <a href="https://www.codecogs.com/eqnedit.php?latex=$\displaystyle\min_{\mathbf{o}_i}&space;\Vert&space;\mathbf{D}\mathbf{o}_i&space;-&space;\mathbf{r}_i\Vert_2^2&space;&plus;&space;k\Vert&space;\mathbf{P}\mathbf{o}_i\Vert_2^2$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\displaystyle\min_{\mathbf{o}_i}&space;\Vert&space;\mathbf{D}\mathbf{o}_i&space;-&space;\mathbf{r}_i\Vert_2^2&space;&plus;&space;k\Vert&space;\mathbf{P}\mathbf{o}_i\Vert_2^2$" title="$\displaystyle\min_{\mathbf{o}_i} \Vert \mathbf{D}\mathbf{o}_i - \mathbf{r}_i\Vert_2^2 + k\Vert \mathbf{P}\mathbf{o}_i\Vert_2^2$" /></a> <br/><br/>
    **Parameters**:

    -   r -- (integer) POD basis size

    -   reg -- (float) L2 regularization parameter. For no
        regularization, set to 0.

    -   xdot -- (r x n_t array) the reduced time derivative data

    -   xhat-- (r x n_t array) the reduced snapshot data

    -   u -- (p x n_t array, optional) the input, if
        `model.input = True`

    **Returns**:
    
    -   None
    

2.  `predict(init, n_timesteps, dt, u = None)`<br/>
    Simulate the learned model with a Runge Kutta scheme<br/>
    **Parameters**:

    -   init -- (r x 1) intial reduced state

    -   n_timesteps -- (int) number of time steps to simulate

    -   dt-- (float) the time step size

    -   u -- (p x n_timesteps array) the input at each
        simulation time step, if
        `model.input = True`

    **Returns**:
    
    -   projected_state -- (r x n_timesteps array) the
        simulated, reduced states<br/>
    -   i -- (int) the time step that the simulation ended on
        (i < n_timesteps only if NaNs occur in simulation)


3.  `get_residual()`<br/>
    Get the residuals of the least squares problem<br/>
    **Parameters**:

    -   None

    **Returns**:
    
    -   residual -- (float) residual of data fit,
        <a href="https://www.codecogs.com/eqnedit.php?latex=\Vert&space;\mathbf{D}\mathbf{O}^T&space;-\dot{&space;\mathbf{X}}^T&space;\Vert_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Vert&space;\mathbf{D}\mathbf{O}^T&space;-\dot{&space;\mathbf{X}}^T&space;\Vert_2^2" title="\Vert \mathbf{D}\mathbf{O}^T -\dot{ \mathbf{X}}^T \Vert_2^2" /></a>
    -   solution -- (float) residual of the solution,
        <a href="https://www.codecogs.com/eqnedit.php?latex=\Vert&space;\mathbf{O}^T&space;\Vert_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Vert&space;\mathbf{O}^T&space;\Vert_2^2" title="\Vert \mathbf{O}^T \Vert_2^2" /></a>


4.  `get_operators()`<br/>
    Get the learned operators<br/>
    **Parameters**:

    -   None

    **Returns**:
    
    -   ops -- (tuple) containing each operator (as an array) as defined
        by `degree` of the model

`opinf_helper.py`
----------------------------------------------------------

Import the opinf helper script as<br/>
`from operator_inference import opinf_helper`.

### functions

The following functions are supported and called as
`opinf_helper.function_name()`.

1.  `normal_equations(D,r,k,num)`<br/>
    Solves the normal equations corresponding to the regularized least
    squares problem <br/>
    <a href="https://www.codecogs.com/eqnedit.php?latex=$\displaystyle\min_{\mathbf{o}_i}&space;\Vert&space;\mathbf{D}\mathbf{o}_i&space;-&space;\mathbf{r}_i\Vert_2^2&space;&plus;&space;k\Vert&space;\mathbf{P}\mathbf{o}_i\Vert_2^2$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\displaystyle\min_{\mathbf{o}_i}&space;\Vert&space;\mathbf{D}\mathbf{o}_i&space;-&space;\mathbf{r}_i\Vert_2^2&space;&plus;&space;k\Vert&space;\mathbf{P}\mathbf{o}_i\Vert_2^2$" title="$\displaystyle\min_{\mathbf{o}_i} \Vert \mathbf{D}\mathbf{o}_i - \mathbf{r}_i\Vert_2^2 + k\Vert \mathbf{P}\mathbf{o}_i\Vert_2^2$" /></a> <br/>
    **Parameters**:

    -   D -- (nd array) data matrix

    -   r -- (nd array) reduced time derivative data

    -   k -- (float) regularization parameter

    -   num -- (int) number of ls problem we are solving [1..r]

    **Returns**:
    
    -   <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{o}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{o}_i" title="\mathbf{o}_i" /></a> -- (nd array) the solution to the least squares
        problem
<br/><br/>
2.  `get_x_sq(X)`<br/>
    Compute squared snapshot data as in [@ben].<br/>
    **Parameters**:

    -   X -- (n_t x r array) reduced snapshot data (transposed)
    
    **Returns**:
    
    -   <a href="https://www.codecogs.com/eqnedit.php?latex=X$^2$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X$^2$" title="X$^2$" /></a> -- (n_t x <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{r(r&plus;1)}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{r(r&plus;1)}{2}" title="\frac{r(r+1)}{2}" /></a> array) reduced snapshot
        data squared without redundant terms.
<br/><br/>
3.  `F2H(F)`<br/>
    Convert quadratic operator <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{F}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{F}$" title="$\mathbf{F}$" /></a> to symmetric quadratic
    operator <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{H}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{H}$" title="$\mathbf{H}$" /></a> for simulating the learned system.<br/>

    **Parameters**:

    -   F -- (r x <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{r(r&plus;1)}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{r(r&plus;1)}{2}" title="\frac{r(r+1)}{2}" /></a> array) learned quadratic
        operator

    **Returns**:
    
    -   H -- (r x r^2 array) symmetric quadratic operator
<br/><br/>

`integration_helpers.py`
----------------------------------------------------------------

Import the integration helper script as<br/>
`from operator_inference import integration_helpers`.

### functions

The following functions are supported and called as
`integration_helpers.function_name()`.
<br/><br/>
1.  `rk4advance_L(x,dt,A,B=0,u=0)`<br/>
    One step of 4th order runge kutta integration of a system of the
    form <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}" /></a>
    <br/>

    **Parameters**:

    -   x -- (r x 1 array) current reduced state

    -   dt -- (float) time step size

    -   A -- (r x r array) linear operator

    -   B -- (r x p array, optional default = 0) input operator
        (only needed if
        `input = True`).

    -   u -- (p x 1 array, optional default = 0) the input at the
        current time step (only needed if
        `input = True`).

    **Returns**:
    
    -   x -- (r x 1 array) reduced state at the next time step
<br/><br/>
2. `rk4advance_Lc(x,dt,A,c,B=0,u=0)`<br/>
    One step of 4th order runge kutta integration of a system of the
    form <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{c}" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}+ \mathbf{c}" /></a> <br/>

    **Parameters**:

    -   x --  (r x 1 array) current reduced state

    -   dt -- (float) time step size

    -   A -- (r x r array) linear operator

    -   c -- (r x 1 array) constant term

    -   B -- (r x p array, optional default = 0) input operator
        (only needed if
        `input = True`).

    -   u -- (p x 1 array, optional default = 0) the input at the
        current time step (only needed if
        `input = True`).

    **Returns**:
    
    -   x -- (r x 1 array) reduced state at the next time step
<br/><br/>
3. `rk4advance_LQ(x,dt,A,H,B=0,u=0)`<br/>
    One step of 4th order runge kutta integration of a system of the
    form
    <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}+ \mathbf{H}(\mathbf{x}\otimes \mathbf{x})" /></a>

    **Parameters**:
    
    -   x -- (r x 1 array) current reduced state

    -   dt -- (float) time step size

    -   A -- (r x r array) linear operator

    -   H -- (r x r^2 array) quadratic operator

    -   B -- (r x p array, optional default = 0) input operator
        (only needed if
        `input = True`).

    -   u -- (p x 1 array, optional default = 0) the input at the
        current time step (only needed if
        `input = True`).

    **Returns**:
    
    -   x -- (r x 1 array) reduced state at the next time step
<br/><br/>
4. `rk4advance_LQc(x,dt,A,H,c,B=0,u=0)`<br/>
    One step of 4th order runge kutta integration of a system of the
    form
    <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})&space;&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{A}\mathbf{x}&plus;&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})&space;&plus;&space;\mathbf{c}" title="\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}+ \mathbf{H}(\mathbf{x}\otimes \mathbf{x}) + \mathbf{c}" /></a>

    **Parameters**:
    
    -   x -- (r x 1 array) current reduced state

    -   dt -- (float) time step size

    -   A -- (r x r array) linear operator

    -   H -- (r x r^2 array) quadratic operator

    -   c -- (r x 1 array) constant term

    -   B -- (r x p array, optional default = 0) input operator
        (only needed if
        `input = True`).

    -   u -- (p x 1 array, optional default = 0) the input at the
        current time step (only needed if
        `input = True`).

    **Returns**:
    
    -   x -- (r x 1 array) reduced state at the next time step
<br/><br/>
5.  `rk4advance_Q(x,dt,H,B=0,u=0)`<br/>
    One step of 4th order runge kutta integration of a system of the
    form <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})" title="\dot{\mathbf{x}} = \mathbf{H}(\mathbf{x}\otimes \mathbf{x})" /></a>

    **Parameters**:
    
    -   x -- (r x 1 array) current reduced state

    -   dt -- (float) time step size

    -   H -- (r x r^2 array) quadratic operator

    -   B -- (r x p array, optional default = 0) input operator
        (only needed if
        `input = True`).

    -   u -- (p x 1 array, optional default = 0) the input at the
        current time step (only needed if
        `input = True`).

    **Returns**:
    
    -   x -- (r x 1 array) reduced state at the next time step
<br/><br/>
6. `rk4advance_Qc(x,dt,H,c,B=0,u=0)`<br/>
    One step of 4th order runge kutta integration of a system of the
    form
    <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{\mathbf{x}}&space;=&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})&space;&plus;&space;\mathbf{c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{\mathbf{x}}&space;=&space;\mathbf{H}(\mathbf{x}\otimes&space;\mathbf{x})&space;&plus;&space;\mathbf{c}" title="\dot{\mathbf{x}} = \mathbf{H}(\mathbf{x}\otimes \mathbf{x}) + \mathbf{c}" /></a>

    **Parameters**:
    
    -   x -- (r x 1 array) current reduced state

    -   dt -- (float) time step size

    -   H -- (r x r^2 array) quadratic operator

    -   c -- (r x 1 array) constant term

    -   B -- (r x p array, optional default = 0) input operator
        (only needed if
        `input = True`).

    -   u -- (p x 1 array, optional default = 0) the input at the
        current time step (only needed if
        `input = True`).<br/>
 
    **Returns**:
    
    -   x -- (r x 1 array) reduced state at the next time step
