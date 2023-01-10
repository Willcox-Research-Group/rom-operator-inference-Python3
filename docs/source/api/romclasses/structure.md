(subsec-romclass-constructor)=
# Defining Model Structure

All ROM classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the right-hand side function $\widehat{\mathbf{F}}$.
Each character in the string corresponds to a single term in the model.

| Character | Name | Continuous Term | Discrete Term |
| :-------- | :--- | :-------------- | :------------ |
| `c` | Constant | $\widehat{\mathbf{c}}$ | $\widehat{\mathbf{c}}$ |
| `A` | Linear | $\widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
| `H` | Quadratic | $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t)]$ | $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$ |
| `G` | Cubic | $\widehat{\mathbf{G}}[\widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t)]$ | $\widehat{\mathbf{G}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$ |
| `B` | Input | $\widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{B}}\mathbf{u}_{j}$ |


<!-- | `C` | Output | $\mathbf{y}(t)=\widehat{C}\widehat{\mathbf{q}}(t)$ | $\mathbf{y}_{k}=\hat{C}\widehat{\mathbf{q}}_{k}$ | -->

The full model form is specified as a single string.

| `modelform` | Continuous ROM Structure | Discrete ROM Structure |
| :---------- | :----------------------- | ---------------------- |
|  `"A"`      | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
|  `"cA"`     | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{c}} + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{c}} + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
|  `"AB"`   | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t) + \widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j} + \widehat{\mathbf{B}}\mathbf{u}_{j}$ |
|  `"HB"`     | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)] + \widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j}] + \widehat{\mathbf{B}}\mathbf{u}_{j}$ |

<!-- | Steady ROM Structure |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{c}} + \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{A}}\widehat{\mathbf{q}} + \widehat{\mathbf{B}}\mathbf{u}$ |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}] + \widehat{\mathbf{B}}\mathbf{u}$ | -->
