## SMC
### **Problem Statement**

We aim to approximate a partially observed matrix $A \in \mathbb{R}^{m \times n}$ using a low-rank decomposition:
$$
\hat{A} = H_1 H_2,
$$
where:
- $H_1 \in \mathbb{R}^{m \times k}$ and $H_2 \in \mathbb{R}^{k \times n}$ are the low-rank factors,
- $k$ is the desired rank of the approximation.

The observation mask $M$ is defined as:
$$
M_{ij} = \begin{cases}
1 & \text{if } A_{ij} \text{ is observed}, \\
0 & \text{otherwise}.
\end{cases}
$$

---

### **Objective Function**

The goal is to minimize the reconstruction loss with regularization:
$$
\mathcal{L}(H_1, H_2) = \frac{1}{2} \| M \odot (A - H_1 H_2) \|_F^2 + \frac{\mu}{2} (\| H_1 \|_F^2 + \| H_2 \|_F^2),
$$
where:
- $\odot$ is the element-wise (Hadamard) product,
- $\| \cdot \|_F$ is the Frobenius norm,
- $\mu > 0$ is a regularization parameter.

---

### **Gradients**

The gradients of the objective function with respect to $H_1$ and $H_2$ are derived as:

#### Gradient with Respect to $H_1$:
$$
\nabla_{H_1} \mathcal{L} = M \odot (H_1 H_2 - A) H_2^T + \mu H_1.
$$

#### Gradient with Respect to $H_2$:
$$
\nabla_{H_2} \mathcal{L} = (M \odot (H_1 H_2 - A))^T H_1 + \mu H_2^T.
$$

---

### **Optimization Algorithm**

#### Combined Variable Representation:
Define $W$, which combines $H_1$ and $H_2$:
$$
W = \begin{bmatrix} H_1 \\ H_2^T \end{bmatrix}.
$$

The combined gradient is:
$$
\nabla \mathcal{L}_W = \begin{bmatrix} \nabla_{H_1} \mathcal{L} \\ \nabla_{H_2} \mathcal{L} \end{bmatrix}.
$$

---

#### Update Rule:
The iterative update for $W$ is:
$$
W_{k+1} = W_k - \alpha_k \nabla \mathcal{L}_W,
$$
where $\alpha_k$ is the step size, determined adaptively.

---

### **Adaptive Step Size**

The step size $\alpha_k$ is determined using the Armijo–Goldstein condition:
$$
\mathcal{L}(W_{k+1}) \leq \mathcal{L}(W_k) + \eta \alpha_k \langle \nabla \mathcal{L}_W, W_{k+1} - W_k \rangle,
$$
where:
- $\eta \in (0, 1)$ is a predefined constant,
- $\langle \cdot, \cdot \rangle$ denotes the inner product.

---

### **Regularization**

#### Fourth-Order Regularization Term:
To improve stability, a higher-order regularization function is introduced:
$$
h(W) = \frac{1}{4} \| W \|_F^4 + \frac{\tau}{2} \| W \|_F^2,
$$
where $\tau > 0$ controls the influence of the higher-order term.

---

#### Difference Measure:
The difference between successive iterates is measured as:
$$
\mathcal{D}_h(W_1, W_2) = h(W_1) - h(W_2) - \langle \nabla h(W_2), W_1 - W_2 \rangle.
$$

---

### **Algorithm Steps**

1. **Initialization:**
   - Randomly initialize $H_1 \in \mathbb{R}^{m \times k}$ and $H_2 \in \mathbb{R}^{k \times n}$.
   - Compute initial loss $\mathcal{L}(H_1, H_2)$.

2. **Iterative Updates:**
   - Compute gradients $\nabla_{H_1} \mathcal{L}$ and $\nabla_{H_2} \mathcal{L}$.
   - Form the combined gradient $\nabla \mathcal{L}_W$.
   - Adjust step size $\alpha_k$ using the Armijo–Goldstein condition.
   - Update $W$ using:
     $$
     W_{k+1} = W_k - \alpha_k \nabla \mathcal{L}_W.
     $$
   - Decompose $W$ back into $H_1$ and $H_2$.

3. **Convergence Check:**
   - Stop if $\| \nabla \mathcal{L}_W \|$ or the change in $\mathcal{L}$ falls below a threshold.

---

### **Outputs**

The algorithm produces:
1. The completed matrix:
   $$
   \hat{A} = H_1 H_2.
   $$
2. Loss history $\mathcal{L}(H_1, H_2)$,
3. RMSE on the test data,
4. Total runtime and number of iterations.
