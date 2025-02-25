## SMC: Structured Matrix Completion Using NEGA2

### **Overview**
NEGA2 is an enhanced Non-Euclidean Gradient Algorithm tailored for solving matrix completion problems.

### **Problem Statement**
The goal is to approximate a partially observed matrix $A \in \mathbb{R}^{m \times n}$ with a low-rank decomposition:
$$
\hat{A} = H_1 H_2,
$$
where $H_1 \in \mathbb{R}^{m \times k}$ and $H_2 \in \mathbb{R}^{k \times n}$, with the observation mask $M$ identifying known entries:
$$
M_{ij} =
\begin{cases}
1 & \text{if } A_{ij} \text{ is observed}, \\
0 & \text{otherwise}.
\end{cases}
$$

### **Objective Function**
The objective is nonconvex:
$$
\mathcal{L}(H_1, H_2) = \frac{1}{2} \| M \odot (A - H_1 H_2) \|_F^2
$$

but $f$ is 3-smooth relative to $h$, a kernel generating distance function:
$$
h(W) = \frac{1}{4} \| W \|_F^4 + \frac{\tau}{2} \| W \|_F^2,
$$
where $W = \begin{bmatrix} H_1 \\ H_2^T \end{bmatrix}$.

### **Reference**
Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). *Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications*. Journal of Machine Learning Research, 23(2022):1-44.
