# Face-Recognition
Recognising faces in images using Python and identifying different persons


## The Eigenface Algorithm

**Input**: -$m$ $n$-dimenstional Datapoints: $m$ faces as imigas with width $w$ and height $h$ ($n=w \cdot h$), in grayscale. We assume that $m\leq n$.
- One $n$-dimensional query-image: Face with width $w$ and height $h$, in grayscale.

Output: The index $i\in \{1, \dots, m\} of the face image $g$, that best matches $g^*$.

### 1. Compute the mean face image
$$g = \frac{1}{m}\sum_{i=1}^m g_i \in R^n$$

### 2. Subtract the mean
$$x_i = g_i.g\in R^n$$

### 3. Setup Datamatrix

$$D = \begin{bmatrix} ───x_1^T ───\\\ \vdots \\\ ───x_m^T ─── \end{bmatrix} \in \mathbb{R}^{m \times n}$$

### 4. Compute the Singular-Value-Decomposition (SVD)

$$X = U \Sigma V^T=U \begin{bmatrix}
\sigma_1 &             \\
         &         \ddots \\
         &        &        \sigma_m
\end{bmatrix}\begin{bmatrix} ───v_1^T ───\\\ \vdots \\\ ───v_m^T ─── \end{bmatrix}$$

- $U \in \mathbb{R}^{m \times m}$ — orthonormal matrix
- $\Sigma \in \mathbb{R}^{m \times m}$ — diagonal matrix with $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_m \geq 0$
- $V \in \mathbb{R}^{n \times m}$ — columns $v_1, \ldots, v_m$ are the singular vectors (eigenfaces)

### 5. Select top-$k$ singular vectors

$$V_k = \begin{bmatrix} 
| & & |\\
v_1 & \cdots & v_k \\
| & & |
\end{bmatrix} \in \mathbb{R}^{n \times k}$$

### 6. Dimensionality reduction (projection)

$$y_i = V_k^T x_i \in \mathbb{R}^k$$

### 7. Project the query image in point eigenbasis

$$y* = V_k^T x_i \in \mathbb{R}^k$$

### 8. Find closest match in original images

$$i = \mathrm{argmin}_{i \in \{1,\ldots,m\}} \lVert y^* - y_i\rVert$$
