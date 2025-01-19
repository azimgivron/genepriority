## SMC: Structured Matrix Completion Using NEGA2

### **Overview**
NEGA2 is an enhanced Non-Euclidean Gradient Algorithm tailored for solving structured matrix completion problems. By leveraging adaptive step sizes and higher-order regularization, it ensures faster convergence and stability for non-convex optimization tasks such as low-rank matrix completion.


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
The objective combines reconstruction loss with regularization:
$$
\mathcal{L}(H_1, H_2) = \frac{1}{2} \| M \odot (A - H_1 H_2) \|_F^2 + \frac{\mu}{2} (\| H_1 \|_F^2 + \| H_2 \|_F^2).
$$

A higher-order regularization term is introduced for improved stability:
$$
h(W) = \frac{1}{4} \| W \|_F^4 + \frac{\tau}{2} \| W \|_F^2,
$$
where $W = \begin{bmatrix} H_1 \\ H_2^T \end{bmatrix}$.


### **NEGA2 Algorithm**
1. **Initialization**: Randomly initialize $H_1$ and $H_2$.
2. **Gradient Computation**: Gradients are computed relative to $H_1$ and $H_2$, incorporating $M$, $A$, and regularization terms.
3. **Adaptive Step Size**: The step size $\alpha_k$ satisfies the Armijo-Goldstein condition to ensure sufficient descent.
4. **Updates**: Iteratively update $W$ using:
   $$
   W_{k+1} = W_k - \alpha_k \nabla \mathcal{L}_W.
   $$
5. **Convergence**: Stop when changes in the objective or gradient norm fall below a threshold.

### **Reference**
Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). *Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications*. Journal of Machine Learning Research, 23(2022):1-44.
