"""Generate Q3 figures for the LaTeX article."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, FastICA

plt.style.use("ggplot")
rng = np.random.default_rng(0)

# --- Sources ---
n = 2000
t = np.linspace(0, 8, n)
x1 = np.sin(2 * t)
x2 = np.sign(np.sin(3 * t))
x3 = signal.sawtooth(2 * np.pi * t)
X = np.vstack([x1, x2, x3])
X = X + 0.02 * rng.standard_normal(X.shape)
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# --- Mix ---
A = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
Y = A @ X

# --- PCA ---
pca = PCA(n_components=3)
Z = pca.fit_transform(Y.T)
Z = Z / np.max(np.abs(Z), axis=0)
Z = Z.T

# --- ICA ---
ica = FastICA(n_components=3, whiten="unit-variance", random_state=0, max_iter=1000)
S = ica.fit_transform(Y.T)
S = S / np.max(np.abs(S), axis=0)
S = S.T

# --- Match ICA components to sources ---
def corr_matrix(R, Src):
    M = np.zeros((R.shape[0], Src.shape[0]))
    for i in range(R.shape[0]):
        for j in range(Src.shape[0]):
            M[i, j] = np.abs(np.corrcoef(R[i], Src[j])[0, 1])
    return M

C_ica = corr_matrix(S, X)
row, col = linear_sum_assignment(-C_ica)
order = col[np.argsort(row)]

names = [r"$x_1$ (sine)", r"$x_2$ (square)", r"$x_3$ (sawtooth)"]
colors_src = ["red", "blue", "purple"]

# Figure 1: four-row summary (sources, mixtures, PCA, ICA)
fig, axs = plt.subplots(4, 3, figsize=(10, 5.5), sharex=True)
row_labels = ["Source", "Mixture", "PCA", "ICA"]
row_colors = [colors_src, ["black"]*3, ["seagreen"]*3, ["darkorange"]*3]
row_data   = [X, Y, Z, S]
col_labels = [names, [r"$y_1$", r"$y_2$", r"$y_3$"],
              [r"PC$_1$", r"PC$_2$", r"PC$_3$"],
              [r"IC$_1$", r"IC$_2$", r"IC$_3$"]]

for r, (data, rcolors, clabels) in enumerate(zip(row_data, row_colors, col_labels)):
    for c in range(3):
        ax = axs[r, c]
        ax.plot(t, data[c], color=rcolors[c], lw=0.8)
        ax.set_ylabel(clabels[c], fontsize=8)
        ax.set_ylim(-4, 4)
        ax.tick_params(labelsize=7)
        if r < 3:
            ax.set_xticklabels([])

for c in range(3):
    axs[-1, c].set_xlabel("time", fontsize=8)

for r, label in enumerate(row_labels):
    axs[r, 0].set_ylabel(f"{label}\n{col_labels[r][0]}", fontsize=8)

plt.tight_layout(pad=0.5)
plt.savefig("fig_q3_rows.pdf", bbox_inches="tight")
plt.close()

# Figure 2: ICA overlay (recovered vs source)
fig, axs = plt.subplots(3, 1, figsize=(9, 4.0), sharex=True)
for k in range(3):
    src = order[k]
    sign = np.sign(np.corrcoef(S[k], X[src])[0, 1])
    axs[k].plot(t, X[src], color="black", lw=2.5, label="source", alpha=0.55)
    axs[k].plot(t, sign * S[k], color="darkorange", lw=1.2, label="ICA")
    axs[k].set_ylabel(names[src], fontsize=8)
    axs[k].set_ylim(-3.5, 3.5)
    axs[k].tick_params(labelsize=7)
    if k == 0:
        axs[k].legend(loc="upper right", fontsize=7)
axs[-1].set_xlabel("time", fontsize=8)
fig.suptitle("Recovered ICs (orange) overlaid on matched sources (black)", fontsize=9)
plt.tight_layout(pad=0.4)
plt.savefig("fig_q3_overlay.pdf", bbox_inches="tight")
plt.close()

# --- Print correlation tables ---
C_pca = corr_matrix(Z, X)
print("PCA |corr| matrix:")
print(np.round(C_pca, 3))
print("\nICA |corr| matrix:")
print(np.round(C_ica, 3))
print("\nMatched ICA |corr|:", np.round(C_ica[np.arange(3), order], 4))
print("ICA->source order:", order)
