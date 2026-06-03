Problem 1

This revised plan specifies the Python functions and classes as used in the **Week 2** (PCA) and **Week 3** (NMF) reference materials.

### **Phase 1: Data Preparation and Baseline**
1.  **Load the Data:** Use `np.loadtxt(..., delimiter=",")` to load `egyptian_goose_clean.csv` and `egyptian_goose_noisy.csv`. 
2.  **Calculate Initial Distance:** To answer the final question, you must first establish the Euclidean distance between the clean image ($X_{clean}$) and the noisy image ($X_{noisy}$). In Python, this can be calculated as `np.sqrt(np.sum((X_clean - X_noisy)**2))`.

### **Phase 2: Principal Component Analysis (Week 2)**
**Principal Component Analysis (PCA)** is used here as a dimensionality reduction method to filter out noise by retaining only the components with the highest variance.

*   **Import:** `from sklearn.decomposition import PCA`.
*   **Iterate over $q$:** For each selected integer $q \in \{1, \dots, 512\}$:
    1.  **Initialize PCA:** Use `PCA(n_components=q)`.
    2.  **Fit and Transform:** Treat the image as a matrix of samples (rows) and call `fit_transform(X_noisy)` to obtain the lower-dimensional representation.
    3.  **Reconstruct:** Use the `inverse_transform(...)` method on the reduced data to map it back to the original 512-dimensional pixel space. This results in the reconstructed image $\hat{X}_{PCA,q}$.
    4.  **Calculate Distance:** Compute the Euclidean distance between $X_{clean}$ and $\hat{X}_{PCA,q}$.

### **Phase 3: Non-negative Matrix Factorisation (Week 3)**
**Non-negative Matrix Factorisation (NMF)** factorises the noisy image into two non-negative matrices $W$ and $H$ such that $X \approx WH$.

*   **Implementation:** The lecture notes provide a custom class, **`NMF_numpy`**, which implements the iterative algorithm to maximise the quasi-likelihood objective $L(W,H)$.
*   **Iterate over $q$:** For each selected integer $q$:
    1.  **Preprocessing:** Ensure the noisy image is strictly non-negative. As seen in the lecture examples, if the image has values near zero, you may need to adjust the pixel values upwards (e.g., `noisy_adj = X_noisy + 1.01`) because the algorithm can have problems with zero values.
    2.  **Initialize and Train:** Create an instance of `NMF_numpy()` and call the `train(noisy_adj, q, n_iter=...)` method.
    3.  **Reconstruct:** The reconstructed image is stored in the `.WH` attribute of the trained NMF object.
    4.  **Post-processing:** If you adjusted the pixel values upward in step 1, remember to subtract that adjustment (e.g., `recons = nmf.WH - 1.01`) before comparing it to the clean image.
    5.  **Calculate Distance:** Compute the Euclidean distance between $X_{clean}$ and the NMF reconstruction.

### **Phase 4: Evaluation and Denoising Analysis**
1.  **Distance Plotting:** Use `plt.plot()` to graph the Euclidean distances for both PCA and NMF against the values of $q$.
2.  **Determine Denoising Capability:** 
    *   Compare these distances to your baseline (Clean vs. Noisy distance).
    *   **Denoising is achieved** if there exists a $q$ where the reconstruction distance is **lower** than the baseline distance.
    *   For PCA, as $q$ approaches 512, the reconstruction will eventually recover the *noisy* image exactly (zero reconstruction error relative to the noisy input), which increases the distance to the *clean* image. Denoising typically happens at an intermediate $q$ where noise-heavy components are discarded.
*   


Problem 2

To tackle Problem 2, you should utilize **Multidimensional Scaling (MDS)**, a dimensionality reduction method covered in **Week 4** that is specifically designed to embed data into a lower-dimensional space while preserving pairwise distances.

### **Phase 1: Data Preparation**
1.  **Construct the Dissimilarity Matrix:** The table provides great-circle distances between six airports. You must first organize this data into a symmetric $6 \times 6$ **dissimilarity matrix** $D$, where $D_{i,j}$ represents the distance between airport $i$ and airport $j$.
2.  **Define Labels:** Create a list of airport codes (CPH, EWR, KEF, LHR, NRT, YVR) to label your map later.

### **Phase 2: Part (a) - Creating the 2D Map (Week 4)**
The objective is to find coordinates in $\mathbb{R}^2$ that minimize the discrepancy between given distances and embedded Euclidean distances.

*   **Function to Use:** Use the `MDS` class from `sklearn.manifold`.
*   **Implementation:**
    1.  **Initialize MDS:** Set `n_components=2` (for $\mathbb{R}^2$) and crucially set `dissimilarity="precomputed"` because you are providing distances directly rather than raw coordinates. 
    2.  **Fit the Model:** Call `fit_transform(D)` on your dissimilarity matrix to obtain the 2D coordinates for each airport.
*   **Visualization:** Use `plt.scatter()` to plot the resulting points. Use `plt.text()` or `ax.annotate()` to label each point with its airport code, as shown in the Week 4 problem solutions.

### **Phase 3: Part (b) - Accuracy and Theoretical Constraints (Week 4)**
1.  **Measure Accuracy:** Accuracy in MDS is typically measured via **Stress** (specifically **Least squares scaling** or Kruskal–Shephard scaling), defined as $S_M = \sum_{i \neq j} (d_{i,j} - \|z_i - z_j\|)^2$. You can access this value in Python using the `.stress_` attribute of the fitted MDS object.
2.  **Explain Imperfection:** Your map cannot be perfectly accurate because the provided distances are **great circle distances**, which the sources define as **geodesic distances on a sphere ($S^2$)**. 
    *   A sphere is a curved 2-manifold that cannot be flattened into a Euclidean plane ($\mathbb{R}^2$) without distorting distances.
    *   MDS seeks a "flat" Euclidean proxy for these "curved" distances, which inherently results in non-zero stress.

### **Phase 4: Part (c) - Increasing Dimensionality (Week 4)**
1.  **Analysis of Accuracy in $\mathbb{R}^d$:** According to the source material, MDS can map samples into any $q \le n$ dimensions.
2.  **Does Accuracy Increase?** 
    *   Generally, increasing $d$ (the dimensionality of the embedding space) provides more degrees of freedom to satisfy the distance constraints, which typically **results in lower stress** (higher accuracy).
    *   However, for this specific case, if the distances are strictly great-circle distances from a sphere, they might still fail to have a perfect Euclidean representation even in higher dimensions, because great-circle distances do not satisfy the properties of Euclidean distance in any dimension.
    *   To verify this computationally, you could repeat the MDS process for $d=3, 4, 5$ and plot the resulting `stress_` values against $d$ to see if and where it plateaus.
*   

Problem 3

To answer Problem 3, you should use the methodologies for **Principal Component Analysis (PCA)** from **Week 2** and **Independent Component Analysis (ICA)** from **Week 4**. This problem is a classic demonstration of the "cocktail-party problem".

### **Phase 1: Construction of Data and Rationale**
To demonstrate the difference between PCA and ICA, your construction must satisfy the fundamental assumptions of ICA while highlighting the limitations of PCA.

1.  **Generate Three Independent Signals ($x_1, x_2, x_3$):** 
    *   **Independence:** The signals must be mutually independent.
    *   **Non-Gaussianity:** According to the **Week 4** slides, ICA requires that at most one of the signals is Gaussian. For a clear demonstration, use three distinctly non-Gaussian distributions (e.g., uniform, Laplace, or a structured periodic signal like a sawtooth wave).
    *   **Rationale for Construction:** PCA is a variance-maximization technique that seeks orthogonal components. If the original source signals are not orthogonal or if the mixtures spread variance in a way that doesn't align with the sources, PCA will fail to separate them. ICA, however, ignores variance and instead **maximizes statistical independence and non-Gaussianity**, allowing it to identify the true underlying source signals.

2.  **Create the Mixtures ($y_1, y_2, y_3$):**
    *   Define a random $3 \times 3$ **mixing matrix** $A$ where $Y = AX$. Ensure the matrix is invertible (full rank) so the signals can theoretically be unmixed.
    *   **Functions to use:** `np.random.uniform` (to generate signals/matrix) and the `@` operator or `np.dot` for the linear mixture.

### **Phase 2: PCA Implementation (Week 2)**
PCA will attempt to separate the signals by finding the directions of maximum variance.

1.  **Preprocessing:** Center the mixture data $Y$ (though the `PCA` class does this automatically).
2.  **Implementation:**
    *   **Function:** Use `sklearn.decomposition.PCA(n_components=3)`.
    *   **Fit and Transform:** Apply `fit_transform(Y.T)` to the transposed mixture matrix (treating time/observations as samples).
3.  **Visualization:** Plot the three resulting principal components. As seen in the **Independent Component Analysis** notebook, these will typically still look like mixtures of the original signals.

### **Phase 3: ICA Implementation (Week 4)**
ICA will attempt to recover the original signals by minimizing mutual information.

1.  **Implementation:**
    *   **Function:** Use `sklearn.decomposition.FastICA(n_components=3, whiten="unit-variance")`.
    *   **Fit and Transform:** Apply `fit_transform(Y.T)` to the mixture data.
2.  **Normalization:** Since the scaling of ICA components is arbitrary, normalize the resulting signals (e.g., `data / np.max(data)`) to match the range of your originals for better comparison.

### **Phase 4: Analysis and Conclusion**
1.  **Comparison:** Plot the original $x$ signals alongside the PCA results and ICA results.
2.  **Evaluate PCA Failure:** Explain that PCA failed because it only looked for orthogonal directions of maximum variance, which does not necessarily correspond to the statistically independent source directions in a mixture.
3.  **Evaluate ICA Success:** Demonstrate that ICA recovered the signals. Note that the recovery is "up to reordering and scaling," which are the known **caveats of ICA** mentioned in the **Week 4** materials (the order of ICs is arbitrary and their variance/sign cannot be recovered).
4.  **Error Measurement:** You can quantitatively show success by calculating the correlation between the original signals and the recovered ICA components to prove they are the same "up to minor error".


Problem 4

To tackle Problem 4, you should focus on the **Curse of Dimensionality** material introduced in **Week 1** and the statistical properties of random vectors discussed throughout the course.

### **Phase 1: Part (a) - Simulation and Empirical Study (Week 1)**
The goal is to demonstrate that as dimensionality $d$ increases, the relative difference between the maximum and minimum distances between random points vanishes.

1.  **Simulation Setup:**
    *   **Function to use:** Use `np.random.randn(10, d)` to generate 10 mutually independent random vectors $X_1, \dots, X_{10}$ from a standard multivariate normal distribution $N_d(0, I_d)$. This is consistent with the probability prerequisites mentioned in **Week 1**.
    *   **Distance Calculation:** For each pair $(i, j)$, calculate the Euclidean distance $\|X_i - X_j\|$. You can use `scipy.spatial.distance.pdist` for efficiency or `np.linalg.norm(Xi - Xj)` for direct calculation.
2.  **Iterative Range of $d$:**
    *   Select a wide range of values for $d$ to see the asymptotic behavior (e.g., $d \in \{2, 10, 100, 1,000, 10,000, 100,000\}$).
3.  **Compute the Ratio $R$:**
    *   For each $d$, identify the maximum and minimum pairwise distances and compute $R = \frac{\max \|X_i - X_j\|}{\min \|X_i - X_j\|}$.
4.  **Visualization:**
    *   Use `plt.plot()` to graph $d$ (log-scale often works best for the x-axis) against the ratio $R$. You should observe that **$R$ converges toward 1** as $d$ becomes very large.

### **Phase 2: Part (b) - Theoretical Explanation (Week 1)**
The theoretical explanation relies on the **Law of Large Numbers (LLN)** applied to the components of the Euclidean distance.

1.  **Distance as a Sum:** Note that the squared Euclidean distance is $\|X_i - X_j\|^2 = \sum_{k=1}^d (X_{i,k} - X_{j,k})^2$.
2.  **Distribution of Components:** Since $X_{i,k}$ and $X_{j,k}$ are i.i.d. $N(0, 1)$, their difference $(X_{i,k} - X_{j,k})$ follows $N(0, 2)$. The squared difference then follows a scaled chi-squared distribution with an expectation $E[(X_{i,k} - X_{j,k})^2] = 2$.
3.  **Application of LLN:** As $d \to \infty$, the sum $\frac{1}{d} \sum_{k=1}^d (X_{i,k} - X_{j,k})^2$ converges in probability to its expectation, which is 2.
4.  **Concentration of Measure:** Consequently, for large $d$, all pairwise distances $\|X_i - X_j\|$ concentrate around the value $\sqrt{2d}$. Since **every pairwise distance converges to the same value**, the ratio of the maximum distance to the minimum distance must converge to 1. This is a "fundamental limit" of mathematics in high dimensions.

### **Phase 3: Part (c) - Relation to the Curse of Dimensionality (Week 1)**
You must relate the convergence $R \to 1$ to the specific problems identified in the **Week 1** lecture notes regarding high-dimensional data.

1.  **Loss of Metric Relevance:** The sources explicitly state that in high dimensions, **"Usual metrics (e.g., Euclidean distance) may lose relevance"**.
2.  **Impact on Algorithms:** If the ratio of the furthest point to the nearest point is approximately 1, the concept of "nearest neighbor" becomes meaningless. This makes distance-based unsupervised learning methods—such as **K-means clustering** (Week 3), **K-medoids** (Week 3), or **MDS** (Week 4)—highly ineffective because the data appears as a "sphere" where every point is equally distant from every other point.
3.  **Rationale for Dimensionality Reduction:** This behavior provides the primary **rationale for dimensionality reduction**: to project data into a lower-dimensional space where distances can once again meaningfully distinguish between samples.


