## Problem definition

The solver solves the following pair of primal and dual problems:

#### Primal:
$$
\begin{align*}
\min &\qquad c^{T}x \\
\text{subject to} &\qquad Ax = b \\
&\qquad Gx + s = h \\
&\qquad s \in \mathcal{K}
\end{align*}
$$

#### Dual:
$$
\begin{align*}
\max &\qquad -b^{T}y - h^{T}z \\
\text{subject to} &\qquad A^{T}y + G^{T}z + c = 0 \\
&\qquad z \in \mathcal{K^{\ast}}
\end{align*}
$$


## Usage notes

- We can mix real and complex cones.
- For complex cone variables, we split the real and imaginary parts as follows: $$split([x_1, x_2, \ldots, x_n]^{T}) = \begin{bmatrix}Re([x_1, x_2, \ldots, x_n]^{T})\\ Im([x_1, x_2, \ldots, x_n]^{T})\end{bmatrix}$$
- For real matrix cones, we vectorize matrices as follows: $$vec(X) = \begin{bmatrix}X_{11} \\ \vdots \\ X_{1n} \\ X_{21} \\ \vdots\end{bmatrix}$$
- For complex matrix cones, we vectorize matrices as follows: $$split(vec(X)) = \begin{bmatrix}Re(vec(X)) \\ Im(vec(X))\end{bmatrix}$$
- The inverse functions $unvec(x)$ and $unsplit(x)$ are also provided.
- The primal variable vector $x$ is real. The primal cost vector $c$ is also real.
- For the linear equality constraints, you just have to separately equate the real and complex parts of your constraints.
- The user has to split/vectorize $G$ and $h$ appropriately. For example, if you have constraints like $$\begin{gather*}F_0 - \sum_{i=1}^{n}{F_i x_i} \in \mathcal{K_1}\\A_0 - \sum_{i=1}^{n}{A_i x_i} \in \mathcal{K_2}\end{gather*}$$ where $\mathcal{K_1}$ is a complex cone and $\mathcal{K_2}$ is a real cone, then the appropriate splitting/vectorization would be $$\begin{gather*}h = \begin{bmatrix}split(vec(F_0)) \\ vec(A_0)\end{bmatrix} \\ G = 
\begin{bmatrix}split(vec(F_1)) & \cdots & split(vec(F_n)) \\ vec(A_1) & \cdots & vec(A_n)\end{bmatrix}\end{gather*}$$

## Input format (not yet implemented)

The input format of a file is given below. `<(type) name: description>` gives the type and description of a particular input. Anything outside `<...>` is a comment and is not part of the input specification.

```
<(Integer) n: Number of primal variables>
<(Integer) p: Number of constraint equations>
<(Integer) k: Number of cones>
<(String) cone_1: Cone 1> <(Any) cone_params_1: Cone 1 parameters>
...
<(String) cone_k: Cone k> <(Any) cone_params_k: Cone k parameters>
┌────────────────────────────────────────────┐
│Let d be the total number of cone variables.│
└────────────────────────────────────────────┘
<(1 x n real vector) c_transpose>
<(p x n real matrix) A>
<(1 x p real vector) b_transpose>
<(d x n real matrix) G>
<(1 x d real vector) h_transpose>
```

## Supported cones:

The list of supported cones and the respective cone parameters are given below. The left hand side of an equality is the format of the cone specification. The right hand side is the cone description. Anything in `[...]` is optional.

- `PSD n [COMPLEX] = Cone of symmetric (Hermitian) n x n positive semidefinite matrices`
- **(UNTESTED)** `DIAGPSD n = Cone of diagonal n x n positive semidefinite matrices`
- **(NOT YET IMPLEMENTED)** `LOGPERSPECEPI n [COMPLEX] = Cone of symmetric (Hermitian) n x n positive semidefinite matrices (T, X, Y) satisfying` $$T \succeq X^{\frac{1}{2}}\log{(X^{\frac{1}{2}}YX^{\frac{1}{2}})}X^{\frac{1}{2}}$$

## Build information:

This solver works with arbitrary precision numbers. You need GMP, Eigen, MPFR C++ to build this.
