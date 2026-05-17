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

We can mix real and complex cones. For complex cone variables, we split the real and imaginary parts as follows:

$$
\text{split}([x_1, x_2, \ldots, x_n]^T) =
\begin{bmatrix}
\text{Re}([x_1, x_2, \ldots, x_n]^T) \\ \text{Im}([x_1, x_2, \ldots, x_n]^T)
\end{bmatrix}
$$

For real symmetric matrix cones, we vectorize matrices as follows:

$$
X = \begin{bmatrix}
X_{11} & \cdots & X_{1n}\\
\vdots & \ddots & \vdots\\
X_{n1} & \cdots & X_{nn}
\end{bmatrix}
$$

$$
\text{vec}(X) =
\begin{bmatrix}
X_{11} & X_{22} & \cdots & X_{nn} & \sqrt{2} X_{21} & \sqrt{2}X_{31} & \sqrt{2}X_{32} &\cdots
\end{bmatrix}^T
$$

For complex Hermitian matrix cones, we vectorize matrices as follows:

$$
\text{split}(\text{vec}(X)) =
\begin{bmatrix}
\text{Re}(\text{vec}(X)) \\ \text{Im}(\text{vec}(X))
\end{bmatrix}
$$

The primal variable vector $x$ and primal cost vector $c$ are real. You have to separately equate the real and complex parts of your linear equality constraints.

You have to split/vectorize your input matrices accordingly. For example, if you have constraints like

$$
\begin{gather*}
F_0 - \sum_{i=1}^{n}{F_i x_i} \in \mathcal{K_1} \\
A_0 - \sum_{i=1}^{n}{A_i x_i} \in \mathcal{K_2}
\end{gather*}
$$

where $\mathcal{K_1}$ is a complex cone and $\mathcal{K_2}$ is a real cone, then the appropriate splitting/vectorization would be

$$
\begin{gather*}
h =
\begin{bmatrix}
\text{split}(\text{vec}(F_0)) \\ \text{vec}(A_0)
\end{bmatrix} \\
G = 
\begin{bmatrix}
\text{split}(\text{vec}(F_1)) & \cdots & \text{split}(\text{vec}(F_n)) \\
\text{vec}(A_1) & \cdots & \text{vec}(A_n)
\end{bmatrix}
\end{gather*}
$$

## Input format

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

The list of supported cones and the respective cone parameters are given below. The left hand side of an equality is the format of the cone specification. The right hand side is the cone description.

- `REALPSD|COMPLEXPSD n`: Cone of symmetric/Hermitian $n \times n$ positive semidefinite matrices
- `NONNEGORTH n`: Cone of non-negative vectors of length $n$
- `REALLPE|COMPLEXLPE n`: Cone of symmetric/Hermitian $n \times n$ positive semidefinite matrices $(T, X, Y)$ satisfying

$$
T \succeq X^{\frac{1}{2}}\log{(X^{\frac{1}{2}}Y^{-1}X^{\frac{1}{2}})}X^{\frac{1}{2}}
$$

## Supported algorithms:

The list of available solver algorithms are given below.

- Nesterov-Todd (symmetric cones only)
- Skajaa-Ye (not implemented)
- Papp-Varga (not implemented)
