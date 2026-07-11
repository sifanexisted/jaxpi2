"""Generate convergence-curve and ablation figures from W&B run histories.

Pulls loss/error histories of the site training runs (project JAXPI-site),
caches them as JSON (gitignored), and renders:

  docs/public/results/<example>_convergence.png   (loss + error vs step)
  docs/public/results/ablation_<study>.png        (overlaid error curves)

    python docs/scripts/gen_convergence.py [--entity sifanw] [--only ex1,..]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "public", "results")
CACHE = os.path.join(os.path.dirname(__file__), "cache")

# example -> run whose convergence curve the example page shows (the
# showcase run; defaults to the baseline when the baseline recipe wins)
EXAMPLES = {
    "advection": "advection__c30_tol0.01",
    "allen_cahn": "allen_cahn__baseline",
    "burgers": "burgers__baseline",
    "inviscid_burgers": "inviscid_burgers__pseudo_time",
    "kdv": "kdv__pseudo_time",
    "ks": "ks__pseudo_time",
    "wave": "wave__baseline",
    "sod_shock_tube": "sod_shock_tube__baseline",
    "lid_driven_cavity": "lid_driven_cavity__pseudo_time_v2",
    "gray_scott": "gray_scott__pt_windows4",
    "ginzburg_landau": "ginzburg_landau__pseudo_time",
    "kolmogorov_flow": "kolmogorov_flow__fixed_pt_windows4_v2",
    "rayleigh_taylor": "rayleigh_taylor__pseudo_time",
    # NOT v3 — that rerun diverged (final error/u = 15.5); v2 is the
    # healthy corrected run the page numbers come from.
    "bfs_flow": "bfs_flow__fixed_pseudo_time_v2",
}

# study -> (title, error metric, [(run suffix, label)])
ABLATIONS = {
    "causal_allen_cahn": (
        "Causal weighting — Allen–Cahn", "error/l2",
        [("allen_cahn__baseline", "causal on (baseline)"),
         ("allen_cahn__no_causal", "causal off")],
    ),
    "causal_ks": (
        "Causal weighting — Kuramoto–Sivashinsky", "error/l2",
        [("ks__baseline", "causal on (baseline)"),
         ("ks__no_causal", "causal off")],
    ),
    "causal_ks_pt": (
        "Causal weighting — KS, long window with pseudo-time", "error/l2",
        [("ks__pseudo_time", "causal on"),
         ("ks__pt_no_causal", "causal off")],
    ),
    "causal_gray_scott_windows": (
        "Causal weighting — Gray–Scott, 4 windows + pseudo-time", "error/v",
        [("gray_scott__pt_windows4", "causal on"),
         ("gray_scott__pt_windows4_no_causal", "causal off")],
    ),
    "pseudo_time_ginzburg_landau": (
        "Pseudo-time stepping — Ginzburg–Landau", "error/u",
        [("ginzburg_landau__baseline", "baseline"),
         ("ginzburg_landau__pseudo_time", "adaptive pseudo-time")],
    ),
    "pseudo_time_inviscid_burgers": (
        "Pseudo-time stepping — inviscid Burgers", "error/l2",
        [("inviscid_burgers__baseline", "baseline"),
         ("inviscid_burgers__pseudo_time", "adaptive pseudo-time"),
         ("inviscid_burgers__fixed_pseudo_time", "fixed pseudo-time")],
    ),
    "pseudo_time_lid_driven_cavity": (
        "Pseudo-time stepping — lid-driven cavity (Re 5000)", "error/l2",
        [("lid_driven_cavity__baseline", "baseline"),
         ("lid_driven_cavity__pseudo_time_v2", "adaptive pseudo-time"),
         ("lid_driven_cavity__fixed_pseudo_time_v2", "fixed pseudo-time")],
    ),
    "pseudo_time_sod_shock_tube": (
        "Pseudo-time stepping — Sod shock tube", "error/p",
        [("sod_shock_tube__baseline", "baseline"),
         ("sod_shock_tube__pseudo_time_v2", "adaptive pseudo-time"),
         ("sod_shock_tube__fixed_pseudo_time_v2", "fixed pseudo-time")],
    ),
    "pseudo_time_kolmogorov_flow": (
        "Pseudo-time stepping — Kolmogorov flow", "error/w",
        [("kolmogorov_flow__baseline", "baseline"),
         ("kolmogorov_flow__pseudo_time_v2", "adaptive pseudo-time")],
    ),
    "pseudo_time_bfs_flow": (
        "Pseudo-time stepping — backward-facing step", "error/u",
        [("bfs_flow__baseline", "baseline"),
         ("bfs_flow__pseudo_time_v2", "adaptive pseudo-time"),
         ("bfs_flow__fixed_pseudo_time_v2", "fixed pseudo-time")],
    ),
    "pseudo_time_ks": (
        "Pseudo-time stepping — Kuramoto–Sivashinsky", "error/l2",
        [("ks__baseline", "baseline"),
         ("ks__pseudo_time", "adaptive pseudo-time")],
    ),
    "arch_burgers": (
        "Architecture — Burgers (parameter-matched, 724k)", "error/l2",
        [("burgers__baseline", "PirateNet (3 blocks)"),
         ("burgers__modified_mlp", "ModifiedMlp (9 layers)"),
         ("burgers__mlp", "Mlp (11 layers)")],
    ),
    "arch_ks": (
        "Architecture — Kuramoto–Sivashinsky (parameter-matched, 724k)", "error/l2",
        [("ks__baseline", "PirateNet (3 blocks)"),
         ("ks__modified_mlp", "ModifiedMlp (9 layers)"),
         ("ks__mlp", "Mlp (11 layers)")],
    ),
    "optimizer_lid_driven_cavity": (
        "Optimizer — lid-driven cavity (Re 5000)", "error/l2",
        [("lid_driven_cavity__baseline", "SOAP"),
         ("lid_driven_cavity__adam", "Adam")],
    ),
    "optimizer_kdv": (
        "Optimizer — Korteweg–de Vries", "error/l2",
        [("kdv__baseline", "SOAP"),
         ("kdv__adam", "Adam")],
    ),
    "weights_wave": (
        "Grad-norm loss balancing — wave", "error/l2",
        [("wave__baseline", "dynamic (grad-norm)"),
         ("wave__constant_weights", "constant weights")],
    ),
    "fourier_advection": (
        "Fourier features — advection (ModifiedMlp)", "error/l2",
        [("advection__mmlp_fourier", "with Fourier features"),
         ("advection__mmlp_no_fourier", "without")],
    ),
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
})

COLORS = ["#3451b2", "#d0342c", "#1a8a5a", "#b57614", "#7b3fa0"]


def fetch_history(api, entity, run_name, refresh=False):
    """Run history (list of dicts), cached under docs/scripts/cache/."""
    os.makedirs(CACHE, exist_ok=True)
    cache_file = os.path.join(CACHE, f"{run_name}.json")
    if os.path.exists(cache_file) and not refresh:
        with open(cache_file) as f:
            return json.load(f)

    runs = api.runs(f"{entity}/JAXPI-site", filters={"display_name": run_name})
    runs = sorted(runs, key=lambda r: r.created_at)
    if not runs:
        return None
    rows = runs[-1].history(samples=2000, pandas=False)
    rows = [
        {k: v for k, v in row.items()
         if k == "_step" or k.startswith(("loss/", "error/"))}
        for row in rows
    ]
    rows.sort(key=lambda r: r["_step"])
    with open(cache_file, "w") as f:
        json.dump(rows, f)
    return rows


def series(rows, key):
    pts = [(r["_step"], r[key]) for r in rows
           if r.get(key) is not None and np.isfinite(r[key])]
    if not pts:
        return None, None
    steps, vals = zip(*pts)
    return np.asarray(steps), np.asarray(vals)


def convergence_figure(name, rows):
    loss_keys = sorted({k for r in rows for k in r if k.startswith("loss/")})
    err_keys = sorted({k for r in rows for k in r if k.startswith("error/")})
    if not loss_keys and not err_keys:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=150, constrained_layout=True)
    for ax, keys, title in [(axes[0], loss_keys, "Training losses"),
                            (axes[1], err_keys, "Relative L2 error")]:
        for i, key in enumerate(keys):
            steps, vals = series(rows, key)
            if steps is None:
                continue
            ax.semilogy(steps, vals, lw=1.4, label=key.split("/", 1)[1],
                        color=COLORS[i % len(COLORS)])
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, which="both")
        if keys:
            ax.legend(frameon=False)
    path = os.path.join(OUT, f"{name}_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.relpath(path, REPO)}")
    return True


def ablation_figure(study, title, metric, arms, histories):
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150, constrained_layout=True)
    plotted = 0
    for i, (run_name, label) in enumerate(arms):
        rows = histories.get(run_name)
        if not rows:
            print(f"  [{study}] missing run {run_name}, skipped arm")
            continue
        key = metric if any(metric in r for r in rows) else None
        if key is None:
            # fall back to the first error metric present
            keys = sorted({k for r in rows for k in r if k.startswith("error/")})
            if not keys:
                continue
            key = keys[0]
        steps, vals = series(rows, key)
        if steps is None:
            continue
        ax.semilogy(steps, vals, lw=1.6, label=label, color=COLORS[i % len(COLORS)])
        plotted += 1
    if plotted < 2:
        plt.close(fig)
        print(f"  [{study}] fewer than 2 arms available, figure skipped")
        return False
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    path = os.path.join(OUT, f"ablation_{study}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.relpath(path, REPO)}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="sifanw")
    parser.add_argument("--only", default=None)
    parser.add_argument("--refresh", action="store_true", help="ignore cache")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    os.makedirs(OUT, exist_ok=True)

    only = args.only.split(",") if args.only else list(EXAMPLES)

    histories = {}

    def get(run_name):
        if run_name not in histories:
            histories[run_name] = fetch_history(api, args.entity, run_name,
                                                refresh=args.refresh)
        return histories[run_name]

    print("convergence curves")
    for name in only:
        rows = get(EXAMPLES.get(name, f"{name}__baseline"))
        if not rows:
            print(f"  {name}: run not found, skipped")
            continue
        convergence_figure(name, rows)

    print("ablation figures")
    for study, (title, metric, arms) in ABLATIONS.items():
        for run_name, _ in arms:
            get(run_name)
        ablation_figure(study, title, metric, arms, histories)


if __name__ == "__main__":
    main()
