"""Generate illustration figures for the Methods pages.

Run from the repo root:

    python docs/scripts/gen_method_figures.py

Outputs to docs/public/methods/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "public", "methods")

INDIGO = "#4f46e5"
CYAN = "#0891b2"
ROSE = "#e11d48"
AMBER = "#d97706"
GRAY = "#64748b"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 170,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"docs/public/methods/{name}")


def causal_weights():
    """Causal gates w_i = exp(-eps * cumsum of earlier residual losses) at
    three stages of training."""
    n = 40
    t = np.linspace(0, 1, n)
    eps = 3.0

    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
    stages = [
        ("early training", 1.2 * np.exp(-((t - 0.15) ** 2) / 0.02) + 0.8),
        ("mid training", 1.5 * np.exp(-((t - 0.55) ** 2) / 0.02) + 0.15 * (t > 0.4)),
        ("late training", 0.06 + 0.04 * np.sin(8 * t) ** 2),
    ]
    for j, (label, loss) in enumerate(stages):
        w = np.exp(-eps * np.concatenate([[0.0], np.cumsum(loss)[:-1]]))
        axes[0, j].plot(t, loss, color=ROSE, lw=2)
        axes[0, j].set_title(label)
        axes[0, j].set_ylim(0, 2.2)
        axes[1, j].plot(t, w, color=INDIGO, lw=2)
        axes[1, j].set_ylim(-0.05, 1.1)
        axes[1, j].set_xlabel("time $t$")
    axes[0, 0].set_ylabel(r"residual $\mathcal{L}_r(t)$")
    axes[1, 0].set_ylabel(r"causal gate $w(t)$")
    fig.suptitle(
        r"$w_i = \exp(-\varepsilon \sum_{k<i} \mathcal{L}_r(t_k))$ — "
        "later times receive weight only once earlier residuals converge",
        y=1.03,
    )
    save(fig, "causal_weights.png")


def loss_balancing():
    """Gradient norms before/after grad-norm weighting."""
    terms = [r"$\mathcal{L}_{ic}$", r"$\mathcal{L}_{bc}$", r"$\mathcal{L}_{r}$"]
    norms = np.array([0.08, 0.5, 12.0])
    lam = norms.sum() / norms
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].bar(terms, norms, color=[GRAY, CYAN, ROSE])
    axes[0].set_yscale("log")
    axes[0].set_title("unweighted: residual dominates")
    axes[0].set_ylabel(r"$\Vert\nabla_\theta \mathcal{L}_i\Vert$")
    axes[1].bar(terms, lam * norms, color=[GRAY, CYAN, ROSE])
    axes[1].set_yscale("log")
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_title(r"weighted: $\Vert\hat\lambda_i \nabla_\theta \mathcal{L}_i\Vert$ equalized")
    save(fig, "loss_balancing.png")


def piratenet_block():
    """Schematic of one PirateNet residual block."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-0.4, 3.4)

    def box(x, y, text, color, w=1.35, h=0.62):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            fc=color, ec="none", alpha=0.14 if color != INDIGO else 0.18,
        ))
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", fc="none", ec=color, lw=1.6,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=10)

    def arrow(x0, y0, x1, y1, color=GRAY, style="-"):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
            color=color, lw=1.5, linestyle=style,
        ))

    y = 1.2
    box(0.1, y, r"$x^{(l)}$", GRAY, w=0.9)
    box(1.6, y, r"$\sigma(W_1 \cdot)$", INDIGO)
    box(3.5, y, r"$f{\odot}U + (1{-}f){\odot}V$", CYAN, w=2.0)
    box(6.1, y, r"$\sigma(W_2 \cdot),\ \mathrm{gate}$", CYAN, w=1.9)
    box(8.55, y, r"$\sigma(W_3 \cdot)$", INDIGO)

    arrow(1.05, y + 0.3, 1.55, y + 0.3)
    arrow(3.05, y + 0.3, 3.45, y + 0.3)
    arrow(5.6, y + 0.3, 6.05, y + 0.3)
    arrow(8.1, y + 0.3, 8.5, y + 0.3)

    # gates U, V from the embeddings
    box(3.9, 2.6, r"$U,\ V$ (from embeddings $\Phi(x)$)", AMBER, w=3.0)
    arrow(4.6, 2.55, 4.45, y + 0.75, color=AMBER)
    arrow(6.6, 2.55, 6.95, y + 0.75, color=AMBER)

    # adaptive skip
    arrow(0.55, y - 0.05, 0.55, 0.25, color=ROSE)
    ax.plot([0.55, 9.7], [0.25, 0.25], color=ROSE, lw=1.5)
    arrow(9.7, 0.25, 9.7, y + 0.2, color=ROSE)
    ax.text(5.1, 0.02, r"adaptive skip:  $x^{(l+1)} = \alpha^{(l)} h^{(l)} + (1-\alpha^{(l)})\, x^{(l)}$,"
                       r"   $\alpha^{(l)}$ trainable, init $\approx 0$",
            ha="center", color=ROSE, fontsize=10)
    ax.text(10.0, y + 0.5, r"$x^{(l+1)}$", color=GRAY, fontsize=11, ha="left")
    save(fig, "piratenet_block.png")


def soap_alignment():
    """GD zigzag vs preconditioned descent on an ill-conditioned quadratic."""
    A = np.diag([1.0, 25.0])

    def path(precond, lr, steps=24):
        x = np.array([-2.4, 0.9])
        xs = [x.copy()]
        for _ in range(steps):
            g = A @ x
            if precond:
                g = np.linalg.solve(A, g)
            x = x - lr * g
            xs.append(x.copy())
        return np.array(xs)

    gd = path(False, lr=0.075)
    newton = path(True, lr=0.6, steps=10)

    xg, yg = np.meshgrid(np.linspace(-2.8, 2.8, 200), np.linspace(-1.2, 1.2, 200))
    z = 0.5 * (A[0, 0] * xg**2 + A[1, 1] * yg**2)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.contour(xg, yg, z, levels=np.geomspace(0.05, 40, 12), colors=GRAY, alpha=0.35, linewidths=1)
    ax.plot(gd[:, 0], gd[:, 1], "o-", ms=3, lw=1.6, color=ROSE, label="first-order (zigzag)")
    ax.plot(newton[:, 0], newton[:, 1], "o-", ms=3, lw=1.8, color=INDIGO,
            label="preconditioned (aligned)")
    ax.plot(0, 0, "*", ms=14, color=AMBER, zorder=5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("ill-conditioned loss: conflicting gradients force zigzag steps")
    save(fig, "soap_alignment.png")


def pseudo_time():
    """Spurious transition layer: pseudo-time update amplifies its residual."""
    t = np.linspace(0, 2.2, 880)[:-40]  # trim FD edge artifact
    h = 0.4
    tau = 1.0
    # Spurious profile for u_t + u = 0 with u(0)=0: true solution u = 0;
    # spurious profile jumps to the decaying branch through a transition layer.
    layer = 0.5 * (1 + np.tanh((t - 0.9) / (h / 4)))
    u = layer * 1.2 * np.exp(-(t - 0.9))
    du = np.gradient(u, t)
    res = du + u
    u_plus = u - tau * res
    res_plus = np.gradient(u_plus, t) + u_plus

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    axes[0].plot(t, np.zeros_like(t), color=GRAY, lw=1.5, label=r"true solution $u \equiv 0$")
    axes[0].plot(t, u, color=ROSE, lw=2, label=r"spurious $u^\dagger$ (layer width $h$)")
    axes[0].plot(t, u_plus, color=INDIGO, lw=2,
                 label=r"$u^{\dagger,+} = u^\dagger - \tau\, \mathcal{R}[u^\dagger]$")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xlabel("$t$")
    axes[0].set_title("one pseudo-time step deforms the profile")

    axes[1].semilogy(t, np.abs(res) + 1e-6, color=ROSE, lw=2, label=r"$|\mathcal{R}[u^\dagger]|$")
    axes[1].semilogy(t, np.abs(res_plus) + 1e-6, color=INDIGO, lw=2,
                     label=r"$|\mathcal{R}[u^{\dagger,+}]|$")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("$t$")
    axes[1].set_title(r"residual defect amplified: $O(h^{-1}) \to O(\tau^2 h^{-3})$")
    save(fig, "pseudo_time.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    causal_weights()
    loss_balancing()
    piratenet_block()
    soap_alignment()
    pseudo_time()
