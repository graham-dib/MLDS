"""Reproduce the Q1 analysis and save figures as vector PDFs for the LaTeX article."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

plt.style.use("ggplot")

X_clean = np.loadtxt("egyptian_goose_clean.csv", delimiter=",")
X_noisy = np.loadtxt("egyptian_goose_noisy.csv", delimiter=",")


def euclid(A, B):
    return np.sqrt(np.sum((A - B) ** 2))


baseline = euclid(X_clean, X_noisy)

# ---- scree ----
scree = PCA().fit(X_noisy)
evr = scree.explained_variance_ratio_
cum = np.cumsum(evr)

fig, axs = plt.subplots(1, 2, figsize=(8, 2.7))
axs[0].bar(range(1, 31), evr[:30] * 100, color="salmon", zorder=2)
axs[0].set_title("Scree plot")
axs[0].set_xlabel("Principal component")
axs[0].set_ylabel("Explained variance (%)")
axs[1].plot(range(1, len(cum) + 1), cum * 100, color="royalblue", zorder=2)
axs[1].set_title("Cumulative explained variance")
axs[1].set_xlabel("Number of components q")
axs[1].set_ylabel("Cumulative (%)")
plt.tight_layout()
plt.savefig("fig_scree.pdf")
plt.close()

q_grid = np.array(sorted(set(
    list(range(1, 21))
    + [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    + [110, 125, 150, 175, 200, 225, 250, 275, 300]
    + [350, 400, 450, 500, 512]
)))

# ---- PCA sweep ----
pca_dist = np.empty(len(q_grid))
pca_recons = {}
for i, q in enumerate(q_grid):
    pca = PCA(n_components=int(q))
    recon = pca.inverse_transform(pca.fit_transform(X_noisy))
    pca_dist[i] = euclid(X_clean, recon)
    pca_recons[int(q)] = recon
best_pca_idx = int(np.argmin(pca_dist))
best_pca_q = int(q_grid[best_pca_idx])


class NMF_numpy:
    def train(self, X: npt.NDArray[np.number], q: int, n_iter: int = 100) -> None:
        d, n = X.shape
        self.W = np.random.uniform(low=0, high=1, size=(d, q)).astype(X.dtype)
        self.H = np.random.uniform(low=0, high=1 / q, size=(q, n)).astype(X.dtype)
        self.WH = self.W @ self.H
        self.logL = np.empty(n_iter, dtype=X.dtype)
        self.dist = np.empty(n_iter, dtype=X.dtype)
        for a in range(n_iter):
            self.W *= ((X / self.WH) @ self.H.T) / self.H.sum(axis=1)[None, :]
            self.WH = self.W @ self.H
            self.H *= (self.W.T @ (X / self.WH)) / self.W.sum(axis=0)[:, None]
            self.WH = self.W @ self.H
            self.logL[a] = np.sum(X * np.log(self.WH) - self.WH)
            self.dist[a] = np.sum((X - self.WH) ** 2)


# ---- NMF convergence ----
np.random.seed(0)
check = NMF_numpy()
check.train(X_noisy, q=20, n_iter=500)
fig, axs = plt.subplots(1, 2, figsize=(8, 2.4))
axs[0].plot(range(1, 501), check.logL, color="red")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel(r"$L(W,H)$")
axs[1].plot(range(1, 501), check.dist, color="blue")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel(r"$\|X-WH\|^2$")
plt.tight_layout()
plt.savefig("fig_conv.pdf")
plt.close()

# ---- NMF sweep ----
np.random.seed(0)
nmf_dist = np.empty(len(q_grid))
nmf_recons = {}
for i, q in enumerate(q_grid):
    nmf = NMF_numpy()
    nmf.train(X_noisy, int(q), n_iter=400)
    nmf_dist[i] = euclid(X_clean, nmf.WH)
    nmf_recons[int(q)] = nmf.WH
best_nmf_idx = int(np.argmin(nmf_dist))
best_nmf_q = int(q_grid[best_nmf_idx])

# ---- distance vs q ----
fig = plt.figure(figsize=(6.2, 3.0))
plt.plot(q_grid, pca_dist, "-o", color="royalblue", markersize=3, label="PCA")
plt.plot(q_grid, nmf_dist, "-s", color="seagreen", markersize=3, label="NMF")
plt.axhline(baseline, color="black", linestyle="--", label="Baseline = %.2f" % baseline)
plt.scatter([best_pca_q], [pca_dist[best_pca_idx]], color="royalblue", s=55, edgecolor="k", zorder=5)
plt.scatter([best_nmf_q], [nmf_dist[best_nmf_idx]], color="seagreen", s=55, edgecolor="k", zorder=5)
plt.xlabel("q")
plt.ylabel("distance to clean image")
plt.legend()
plt.tight_layout()
plt.savefig("fig_curve.pdf")
plt.close()

# ---- image comparison ----
fig, axs = plt.subplots(1, 4, figsize=(10, 2.7))
panels = [("Clean", X_clean), ("Noisy (d=%.2f)" % baseline, X_noisy),
          ("PCA q=%d (d=%.2f)" % (best_pca_q, pca_dist[best_pca_idx]), pca_recons[best_pca_q]),
          ("NMF q=%d (d=%.2f)" % (best_nmf_q, nmf_dist[best_nmf_idx]), nmf_recons[best_nmf_q])]
for ax, (t, img) in zip(axs, panels):
    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=X_clean.min(), vmax=X_clean.max())
    ax.set_title(t, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("fig_compare.png", dpi=200)  # png: photographic content, smaller than vector
plt.close()

# ---- numbers for the article ----
print("baseline %.4f" % baseline)
print("PC1 %.1f PC2 %.1f q80 %d q95 %d"
      % (evr[0]*100, evr[1]*100, np.searchsorted(cum, 0.80)+1, np.searchsorted(cum, 0.95)+1))
print("PCA best q=%d d=%.4f cum=%.1f"
      % (best_pca_q, pca_dist[best_pca_idx], cum[best_pca_q-1]*100))
print("NMF best q=%d d=%.4f cum=%.1f"
      % (best_nmf_q, nmf_dist[best_nmf_idx], cum[best_nmf_q-1]*100))
print("PCA q=512 d=%.4f" % pca_dist[-1])
print("PCA beats:", q_grid[pca_dist < baseline].min(), "..", q_grid[pca_dist < baseline].max())
print("NMF beats:", q_grid[nmf_dist < baseline].min(), "..", q_grid[nmf_dist < baseline].max())
