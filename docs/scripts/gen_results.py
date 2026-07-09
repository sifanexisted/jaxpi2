"""Generate PINN result figures for the docs website from trained checkpoints.

For each example, restores the trained model from its checkpoint, predicts the
solution on the reference grid, and renders a reference / prediction / error
figure into docs/public/results/, plus a JSON file with the final relative L2
errors used by the docs pages.

    python docs/scripts/gen_results.py [--ckpt-root PATH] [--only ex1,ex2]

Checkpoints are expected under <ckpt-root>/<example>__baseline/ckpt (the
layout produced by training with --config.saving.ckpt_path=<ckpt-root> and
--config.wandb.name=<example>__baseline).
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "public", "results")
DEFAULT_CKPT_ROOT = "/data/sifanw/pinns/site"

sys.path.insert(0, REPO)

import jax.numpy as jnp
from jax import vmap

from jaxpi.checkpointing import create_checkpoint_manager, restore_checkpoint
from jaxpi.models import create_model
from jaxpi.utils import get_eval_params


@contextmanager
def example(name):
    """Temporarily chdir into an example and make its modules importable."""
    path = os.path.join(REPO, "examples", name)
    old_cwd = os.getcwd()
    sys.path.insert(0, path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)
        for mod in ("utils", "models", "configs", "configs.base", "configs.baseline"):
            sys.modules.pop(mod, None)


def load_config(name, config_file="baseline"):
    with example(name):
        import importlib

        module = importlib.import_module(f"configs.{config_file}")
        return module.get_config()


def restore(config, model, ckpt_root, run_name, windowed=False):
    path = os.path.join(ckpt_root, run_name, "ckpt")
    suffix = "time_window_1" if windowed else None
    mngr = create_checkpoint_manager(config.saving, path, suffix=suffix)
    model.state = restore_checkpoint(mngr, model.state)
    return get_eval_params(model.state, config.optim.schedule_free)


def rel_l2(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def panels_1d(name, u_ref, u_pred, t, x, cmap, xlabel="t", ylabel="x"):
    """Space-time panels: reference / prediction / abs error."""
    u_ref, u_pred = np.asarray(u_ref), np.asarray(u_pred)
    err = np.abs(u_pred - u_ref)
    vmax = np.percentile(np.abs(u_ref), 99.9)
    extent = [float(t[0]), float(t[-1]), float(x[0]), float(x[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), dpi=150, constrained_layout=True)
    for ax, field, title, kw in [
        (axes[0], u_ref, "Reference", {"vmin": -vmax, "vmax": vmax, "cmap": cmap}),
        (axes[1], u_pred, "PINN prediction", {"vmin": -vmax, "vmax": vmax, "cmap": cmap}),
        (axes[2], err, "Absolute error", {"cmap": "magma"}),
    ]:
        im = ax.imshow(field.T, origin="lower", aspect="auto", extent=extent, **kw)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, shrink=0.85)
    _save(fig, name)


def panels_2d(name, ref, pred, cmap, vsym=True, labels=("x", "y")):
    """2D snapshot panels: reference / prediction / abs error."""
    ref, pred = np.asarray(ref), np.asarray(pred)
    err = np.abs(pred - ref)
    kw = {"cmap": cmap}
    if vsym:
        vmax = np.percentile(np.abs(ref), 99.9)
        kw.update(vmin=-vmax, vmax=vmax)
    else:
        kw.update(vmin=np.percentile(ref, 0.1), vmax=np.percentile(ref, 99.9))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=150, constrained_layout=True)
    for ax, field, title, k in [
        (axes[0], ref, "Reference", kw),
        (axes[1], pred, "PINN prediction", kw),
        (axes[2], err, "Absolute error", {"cmap": "magma"}),
    ]:
        im = ax.imshow(field.T, origin="lower", aspect="equal", **k)
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85)
    _save(fig, name)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}_pred.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.relpath(path, REPO)}")


def predict_1d(model, params, t_star, x_star):
    return vmap(
        vmap(model.neural_net, (None, None, 0)), (None, 0, None)
    )(params, t_star, x_star)


def predict_snapshots(fn, params, t_star, coords):
    """Loop over time to keep memory bounded.

    `fn` is a doubly-vmapped pred_fn (outer axis: time), so each call gets a
    length-1 time array and returns shape (1, N).
    """
    outs = []
    for t in np.asarray(t_star):
        out = fn(params, jnp.array([float(t)]), coords[:, 0], coords[:, 1])
        outs.append(np.asarray(out)[0])
    return np.stack(outs)


# ---------------------------------------------------------------------------
# per-example generators; each returns {"metric_name": value}
# ---------------------------------------------------------------------------


def gen_simple_1d(name, model_cls_name, cmap, run=None, config_file="baseline",
                  model_kwargs=None, ckpt_root=DEFAULT_CKPT_ROOT):
    """Examples following the (u0, t_star, x_star) pattern."""
    run = run or f"{name}__baseline"
    config = load_config(name, config_file)
    with example(name):
        import models
        import utils

        u_ref, t_star, x_star = utils.get_dataset()
        kwargs = dict(u0=u_ref[0, :], t_star=t_star, x_star=x_star)
        kwargs.update(model_kwargs or {})
        model = create_model(config, getattr(models, model_cls_name), **kwargs)
        params = restore(config, model, ckpt_root, run)
        u_pred = predict_1d(model, params, t_star, x_star)

    error = rel_l2(u_pred, u_ref)
    panels_1d(name, u_ref, u_pred, t_star, x_star, cmap)
    return {"l2_error": error}


def gen_advection(ckpt_root):
    return gen_simple_1d("advection", "Advection1D", "twilight_shifted",
                         model_kwargs={"c": 50}, ckpt_root=ckpt_root)


def gen_allen_cahn(ckpt_root):
    return gen_simple_1d("allen_cahn", "AllenCahn", "Spectral_r", ckpt_root=ckpt_root)


def gen_burgers(ckpt_root):
    return gen_simple_1d("burgers", "Burgers", "coolwarm", ckpt_root=ckpt_root)


def gen_inviscid_burgers(ckpt_root):
    return gen_simple_1d("inviscid_burgers", "InviscidBurgers", "coolwarm",
                         ckpt_root=ckpt_root)


def gen_kdv(ckpt_root):
    return gen_simple_1d("kdv", "KDV", "turbo", ckpt_root=ckpt_root)


def gen_wave(ckpt_root):
    return gen_simple_1d("wave", "Wave1D", "RdBu_r", model_kwargs={"c": 2.0},
                         ckpt_root=ckpt_root)


def gen_ks(ckpt_root):
    name, run = "ks", "ks__baseline"
    config = load_config(name)
    with example(name):
        import models
        import utils

        u_ref, t_ref, x_star = utils.get_dataset(time_range=config.time_range)
        # Mirror main.py's window slicing (num_time_windows = 1)
        num_time_steps = len(t_ref) // config.training.num_time_windows
        t_star = t_ref[:num_time_steps]
        model = create_model(config, models.KS, u0=u_ref[0, :], t_star=t_star, x_star=x_star)
        params = restore(config, model, ckpt_root, run, windowed=True)
        u_pred = predict_1d(model, params, t_star, x_star)

    u_ref = u_ref[:num_time_steps]
    error = rel_l2(u_pred, u_ref)
    panels_1d(name, u_ref, u_pred, t_star, x_star, "inferno")
    return {"l2_error": error}


def gen_sod_shock_tube(ckpt_root):
    name, run = "sod_shock_tube", "sod_shock_tube__baseline"
    config = load_config(name)
    with example(name):
        import models
        import utils

        (rho_ref, u_ref, p_ref, T, X, t_star, x_star,
         left_coords, right_coords) = utils.get_dataset()
        model = create_model(
            config, models.Euler1D,
            rho0=rho_ref[:, 0], u0=u_ref[:, 0], p0=p_ref[:, 0],
            t_star=t_star, x_star=x_star,
            left_coords=left_coords, right_coords=right_coords,
        )
        params = restore(config, model, ckpt_root, run)
        rho_pred, u_pred, p_pred = vmap(
            vmap(model.neural_net, (None, 0, None)), (None, None, 0)
        )(params, t_star, x_star)  # (Nx, Nt)? mirror eval: outer over x

    # reference arrays are (Nx, Nt); predictions above are (Nx, Nt) as well
    errors = {
        "rho_error": rel_l2(rho_pred, rho_ref),
        "u_error": rel_l2(u_pred, u_ref),
        "p_error": rel_l2(p_pred, p_ref),
    }
    # panels: use density (transpose to (t, x) for the shared 1D renderer)
    panels_1d(name, np.asarray(rho_ref).T, np.asarray(rho_pred).T,
              t_star, x_star, "cividis")
    return errors


def gen_lid_driven_cavity(ckpt_root):
    name, run = "lid_driven_cavity", "lid_driven_cavity__baseline"
    config = load_config(name)
    with example(name):
        import models
        import utils

        u_ref, v_ref, x_star, y_star, _ = utils.get_dataset(config.Re)
        model = create_model(config, models.NavierStokes2D, nu=1 / config.Re)
        params = restore(config, model, ckpt_root, run)
        u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star)
        v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star)

    U_ref = np.sqrt(np.asarray(u_ref) ** 2 + np.asarray(v_ref) ** 2)
    U_pred = np.sqrt(np.asarray(u_pred) ** 2 + np.asarray(v_pred) ** 2)
    error = rel_l2(U_pred, U_ref)
    panels_2d(name, U_ref, U_pred, "viridis", vsym=False)
    return {"l2_error": error}


def gen_gray_scott(ckpt_root):
    return _gen_reaction_diffusion(
        "gray_scott", "GrayScott",
        param_names=("b1", "b2", "c1", "c2", "eps1", "eps2"),
        cmap="magma", vsym=False, field_idx=1, ckpt_root=ckpt_root,
    )


def gen_ginzburg_landau(ckpt_root):
    return _gen_reaction_diffusion(
        "ginzburg_landau", "GinzburgLandau",
        param_names=("eps", "k"),
        cmap="twilight", vsym=True, field_idx=0, ckpt_root=ckpt_root,
    )


def _gen_reaction_diffusion(name, cls_name, param_names, cmap, vsym, field_idx,
                            ckpt_root):
    """gray_scott / ginzburg_landau: (u, v) systems on a 2D periodic grid."""
    run = f"{name}__baseline"
    config = load_config(name)
    with example(name):
        import models
        import utils

        data = utils.get_dataset(time_range=config.time_range)
        u_ref, v_ref, t_ref, x_star, y_star = data[:5]
        pde_params = dict(zip(param_names, data[5:]))

        num_time_steps = len(t_ref) // config.training.num_time_windows
        t_star = t_ref[:num_time_steps]
        dt = t_star[1] - t_star[0]
        t1 = t_star[-1] + 1.1 * dt

        nx, ny = u_ref.shape[1], u_ref.shape[2]
        XX, YY = jnp.meshgrid(jnp.asarray(x_star), jnp.asarray(y_star), indexing="ij")
        coords = jnp.stack([XX.ravel(), YY.ravel()], axis=1)

        model = create_model(config, getattr(models, cls_name), t_max=t1, **pde_params)
        params = restore(config, model, ckpt_root, run, windowed=True)

        fn = [model.u_pred_fn, model.v_pred_fn][field_idx]
        preds = predict_snapshots(fn, params, t_star, coords).reshape(-1, nx, ny)

    ref = np.asarray([u_ref, v_ref][field_idx])[:num_time_steps]
    error = rel_l2(preds, ref)
    panels_2d(name, ref[-1], preds[-1], cmap, vsym=vsym)
    return {"l2_error": error}


def gen_kolmogorov_flow(ckpt_root):
    name, run = "kolmogorov_flow", "kolmogorov_flow__baseline"
    config = load_config(name)
    with example(name):
        import models
        import utils

        u_ref, v_ref, w_ref, t_ref, coords, nu = utils.get_dataset(
            time_range=config.time_range)
        num_time_steps = len(t_ref) // config.training.num_time_windows
        t_star = t_ref[:num_time_steps]
        dt = t_star[1] - t_star[0]
        t1 = t_star[-1] + 1.1 * dt

        model = create_model(config, models.NavierStokes2D, t_max=t1, nu=nu)
        params = restore(config, model, ckpt_root, run, windowed=True)

        w_pred = predict_snapshots(
            model.w_pred_fn, params, t_star, jnp.asarray(coords)
        )

    n = int(round(np.sqrt(w_ref.shape[1])))
    w_ref = np.asarray(w_ref)[:num_time_steps]
    error = rel_l2(w_pred, w_ref)
    panels_2d(name, w_ref[-1].reshape(n, n), w_pred[-1].reshape(n, n), "RdBu_r")
    return {"w_error": error}


def gen_bfs_flow(ckpt_root):
    name, run = "bfs_flow", "bfs_flow__baseline"
    config = load_config(name)
    with example(name):
        import models
        import utils

        (u_ref, v_ref, p_ref, coords, inflow_coords, outflow_coords,
         wall_coords, nu) = utils.get_dataset()
        u_inflow, _ = utils.inflow_profile(inflow_coords[:, 1])
        model = create_model(
            config, models.NavierStokes2D,
            u_inflow=u_inflow, inflow_coords=inflow_coords,
            outflow_coords=outflow_coords, wall_coords=wall_coords, nu=nu,
        )
        params = restore(config, model, ckpt_root, run)
        preds = vmap(model.neural_net, (None, 0, 0))(
            params, coords[:, 0], coords[:, 1])
        u_pred, v_pred = preds[0], preds[1]

    U_ref = np.sqrt(np.asarray(u_ref) ** 2 + np.asarray(v_ref) ** 2)
    U_pred = np.sqrt(np.asarray(u_pred) ** 2 + np.asarray(v_pred) ** 2)
    error = rel_l2(U_pred, U_ref)

    coords = np.asarray(coords)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.6), dpi=150, constrained_layout=True)
    for ax, field, title, cmap in [
        (axes[0], U_ref, "Reference", "viridis"),
        (axes[1], U_pred, "PINN prediction", "viridis"),
        (axes[2], np.abs(U_pred - U_ref), "Absolute error", "magma"),
    ]:
        im = ax.tricontourf(coords[:, 0], coords[:, 1], field, levels=100, cmap=cmap)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.9)
    _save(fig, name)
    return {"l2_error": error}


GENERATORS = {
    "advection": gen_advection,
    "allen_cahn": gen_allen_cahn,
    "burgers": gen_burgers,
    "inviscid_burgers": gen_inviscid_burgers,
    "kdv": gen_kdv,
    "wave": gen_wave,
    "ks": gen_ks,
    "sod_shock_tube": gen_sod_shock_tube,
    "lid_driven_cavity": gen_lid_driven_cavity,
    "gray_scott": gen_gray_scott,
    "ginzburg_landau": gen_ginzburg_landau,
    "kolmogorov_flow": gen_kolmogorov_flow,
    "bfs_flow": gen_bfs_flow,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--only", default=None, help="comma-separated example names")
    args = parser.parse_args()

    only = args.only.split(",") if args.only else list(GENERATORS)
    os.makedirs(OUT, exist_ok=True)

    metrics_path = os.path.join(OUT, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    for name in only:
        print(name)
        try:
            metrics[name] = {k: round(v, 6) for k, v in GENERATORS[name](args.ckpt_root).items()}
            print(f"  errors: {metrics[name]}")
        except FileNotFoundError as e:
            print(f"  skipped (no checkpoint): {e}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"\nWrote {os.path.relpath(metrics_path, REPO)}")


if __name__ == "__main__":
    main()
