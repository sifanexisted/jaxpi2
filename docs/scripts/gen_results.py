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

# Which W&B/checkpoint run supplies each example's showcase figure;
# overridable with --runs (defaults to <example>__baseline).
# Note: both no-transfer runs reuse a window-1 checkpoint copied from a
# recipe-identical earlier run (window 1 always trains from scratch, so it is
# the same under either transfer-learning setting).
RUN_OVERRIDES = {
    "gray_scott": "gray_scott__pt_windows4_no_transfer",
    "rayleigh_taylor": "rayleigh_taylor__full_traj_5win_noTL",
}

# Number of time windows the showcase run was trained with (overridable with
# --windows); windowed examples default to their config value.
WINDOW_OVERRIDES = {
    "gray_scott": 4,
    "rayleigh_taylor": 5,
}

# time_range the showcase run was trained with, when it differs from the
# config default (overridable with --time-ranges, e.g. gray_scott=0:1.0).
TIME_RANGE_OVERRIDES = {
    "gray_scott": (0.0, 1.0),
    "rayleigh_taylor": (0.0, 9.75),
}


def run_for(name):
    return RUN_OVERRIDES.get(name, f"{name}__baseline")


def windows_for(name, config):
    return int(WINDOW_OVERRIDES.get(name, config.training.num_time_windows))

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
        config = module.get_config()
    if name in TIME_RANGE_OVERRIDES:
        config.time_range = TIME_RANGE_OVERRIDES[name]
    return config


def restore(config, model, ckpt_root, run_name, windowed=False, window=1):
    path = os.path.join(ckpt_root, run_name, "ckpt")
    suffix = f"time_window_{window}" if windowed else None
    mngr = create_checkpoint_manager(config.saving, path, suffix=suffix)
    try:
        model.state = restore_checkpoint(mngr, model.state)
    except ValueError:
        # Aux state (e.g. loss/pts weight keys) may have been renamed since
        # the run was trained; restore with the on-disk structure instead and
        # keep only the parameters, which is all the figures need.
        restored = mngr.restore(mngr.latest_step())
        return restored["params"]
    # Use the raw training iterates, matching what the example evaluators log
    # during training (empirically better than the schedule-free averages at
    # the end of these runs, where the LR has fully decayed).
    return model.state.params


def rel_l2(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


# ---------------------------------------------------------------------------
# unified figure style (shared by every static figure and animation)
# ---------------------------------------------------------------------------

INK = "#1e293b"       # slate-800, matches the site's text color
MUTED = "#64748b"     # slate-500
BRAND = "#4f46e5"     # site brand indigo
ERR_CMAP = "magma"    # error panel colormap, unified across examples

FONT_DIR = "/root/code/sifanw/site_experiments/fonts/inter_extract/extras/ttf"


def _register_fonts():
    """Use Inter (the docs site font) when available, DejaVu otherwise."""
    from matplotlib import font_manager

    if os.path.isdir(FONT_DIR):
        for fname in ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf"):
            path = os.path.join(FONT_DIR, fname)
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
        if any("Inter" == f.name for f in font_manager.fontManager.ttflist):
            return "Inter"
    return "DejaVu Sans"


plt.rcParams.update({
    "font.family": _register_fonts(),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": INK,
    "axes.edgecolor": "#e2e8f0",
    "axes.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.titlecolor": INK,
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.labelcolor": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 150,
})

PANEL_TITLES = ("Reference", "PINN prediction", "Absolute error")


def _style_colorbar(cbar):
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.5, color=MUTED, width=0.8, length=3)


def panels_1d(name, u_ref, u_pred, t, x, cmap, xlabel="t", ylabel="x"):
    """Space-time panels: reference / prediction / abs error."""
    u_ref, u_pred = np.asarray(u_ref), np.asarray(u_pred)
    err = np.abs(u_pred - u_ref)
    vmax = np.percentile(np.abs(u_ref), 99.9)
    extent = [float(t[0]), float(t[-1]), float(x[0]), float(x[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)
    fields = (u_ref, u_pred, err)
    kws = (
        {"vmin": -vmax, "vmax": vmax, "cmap": cmap},
        {"vmin": -vmax, "vmax": vmax, "cmap": cmap},
        {"cmap": ERR_CMAP},
    )
    for ax, field, title, kw in zip(axes, fields, PANEL_TITLES, kws):
        im = ax.imshow(field.T, origin="lower", aspect="auto", extent=extent, **kw)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _style_colorbar(fig.colorbar(im, ax=ax, shrink=0.85))
    _save(fig, name)


def _panel_limits(ref, vsym):
    if vsym:
        vmax = np.percentile(np.abs(ref), 99.9)
        return {"vmin": -vmax, "vmax": vmax}
    return {"vmin": float(np.percentile(ref, 0.1)),
            "vmax": float(np.percentile(ref, 99.9))}


def panels_2d(name, ref, pred, cmap, vsym=True, labels=("x", "y")):
    """2D snapshot panels: reference / prediction / abs error."""
    ref, pred = np.asarray(ref), np.asarray(pred)
    err = np.abs(pred - ref)
    kw = {"cmap": cmap, **_panel_limits(ref, vsym)}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    fields = (ref, pred, err)
    kws = (kw, kw, {"cmap": ERR_CMAP})
    for ax, field, title, k in zip(axes, fields, PANEL_TITLES, kws):
        im = ax.imshow(field.T, origin="lower", aspect="equal", **k)
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xticks([])
        ax.set_yticks([])
        _style_colorbar(fig.colorbar(im, ax=ax, shrink=0.85))
    _save(fig, name)


def animate_2d(name, ref, pred, t_star, cmap, vsym=True, fps=10, max_frames=100):
    """Animated reference / prediction / error panels for 2D time-dependent
    problems: (T, nx, ny) arrays -> docs/public/results/<name>_pred.mp4."""
    import imageio.v2 as imageio

    ref, pred = np.asarray(ref), np.asarray(pred)
    stride = max(1, len(ref) // max_frames)
    ref_s, pred_s, t_s = ref[::stride], pred[::stride], np.asarray(t_star)[::stride]
    err_s = np.abs(pred_s - ref_s)

    kw = {"cmap": cmap, **_panel_limits(ref, vsym)}
    err_kw = {"cmap": ERR_CMAP, "vmin": 0.0,
              "vmax": float(np.percentile(err_s, 99.5))}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    images = []
    for ax, field, title, k in zip(
        axes, (ref_s[0], pred_s[0], err_s[0]), PANEL_TITLES, (kw, kw, err_kw)
    ):
        images.append(ax.imshow(field.T, origin="lower", aspect="equal", **k))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        _style_colorbar(fig.colorbar(images[-1], ax=ax, shrink=0.85))
    # time badge inside the reference panel (readable on any colormap)
    time_text = axes[0].text(
        0.04, 0.955, "", transform=axes[0].transAxes, ha="left", va="top",
        fontsize=9, color=INK, weight="medium",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}_pred.mp4")
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    fig.canvas.draw()
    for i in range(len(ref_s)):
        for im, field in zip(images, (ref_s[i], pred_s[i], err_s[i])):
            im.set_data(field.T)
        time_text.set_text(f"t = {float(t_s[i]):.3f}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    print(f"  {os.path.relpath(path, REPO)} ({os.path.getsize(path) // 1024} KB)")


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
    run = run or run_for(name)
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
    name, run = "ks", run_for("ks")
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
    name, run = "sod_shock_tube", run_for("sod_shock_tube")
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
    name, run = "lid_driven_cavity", run_for("lid_driven_cavity")
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
    run = run_for(name)
    config = load_config(name)
    with example(name):
        import models
        import utils

        data = utils.get_dataset(time_range=config.time_range)
        u_ref, v_ref, t_ref, x_star, y_star = data[:5]
        pde_params = dict(zip(param_names, data[5:]))

        windows = windows_for(name, config)
        num_time_steps = len(t_ref) // windows
        t_star = t_ref[:num_time_steps]  # every window trains on this local grid
        dt = t_star[1] - t_star[0]
        t1 = t_star[-1] + 1.1 * dt

        nx, ny = u_ref.shape[1], u_ref.shape[2]
        XX, YY = jnp.meshgrid(jnp.asarray(x_star), jnp.asarray(y_star), indexing="ij")
        coords = jnp.stack([XX.ravel(), YY.ravel()], axis=1)

        model = create_model(config, getattr(models, cls_name), t_max=t1, **pde_params)
        fn = [model.u_pred_fn, model.v_pred_fn][field_idx]

        # Stitch predictions from each window's checkpoint
        preds = []
        for k in range(windows):
            params = restore(config, model, ckpt_root, run, windowed=True, window=k + 1)
            preds.append(predict_snapshots(fn, params, t_star, coords))
        preds = np.concatenate(preds).reshape(-1, nx, ny)

    ref = np.asarray([u_ref, v_ref][field_idx])[: windows * num_time_steps]
    error = rel_l2(preds, ref)
    panels_2d(name, ref[-1], preds[-1], cmap, vsym=vsym)
    animate_2d(name, ref, preds, t_ref[: windows * num_time_steps], cmap, vsym=vsym)
    return {"l2_error": error}


def gen_kolmogorov_flow(ckpt_root):
    name, run = "kolmogorov_flow", run_for("kolmogorov_flow")
    config = load_config(name)
    with example(name):
        import models
        import utils

        u_ref, v_ref, w_ref, t_ref, coords, nu = utils.get_dataset(
            time_range=config.time_range)
        windows = windows_for(name, config)
        num_time_steps = len(t_ref) // windows
        t_star = t_ref[:num_time_steps]
        dt = t_star[1] - t_star[0]
        t1 = t_star[-1] + 1.1 * dt

        model = create_model(config, models.NavierStokes2D, t_max=t1, nu=nu)

        w_pred = []
        for k in range(windows):
            params = restore(config, model, ckpt_root, run, windowed=True, window=k + 1)
            w_pred.append(
                predict_snapshots(model.w_pred_fn, params, t_star, jnp.asarray(coords))
            )
        w_pred = np.concatenate(w_pred)

    n = int(round(np.sqrt(w_ref.shape[1])))
    w_ref = np.asarray(w_ref)[: windows * num_time_steps]
    error = rel_l2(w_pred, w_ref)
    panels_2d(name, w_ref[-1].reshape(n, n), w_pred[-1].reshape(n, n), "RdBu_r")
    animate_2d(
        name, w_ref.reshape(-1, n, n), w_pred.reshape(-1, n, n),
        t_ref[: windows * num_time_steps], "RdBu_r",
    )
    return {"w_error": error}


def gen_rayleigh_taylor(ckpt_root):
    name, run = "rayleigh_taylor", run_for("rayleigh_taylor")
    config = load_config(name)
    with example(name):
        import models
        import utils

        (uv_ref, p_ref, temp_ref, t_ref, mesh,
         alpha1, alpha2, alpha3, alpha4, Ra, Pr, Ge) = utils.get_dataset(
            time_range=config.time_range)

        windows = windows_for(name, config)
        num_time_steps = len(t_ref) // windows
        t_star = t_ref[:num_time_steps]
        dt = t_star[1] - t_star[0]
        t1 = t_star[-1] + 1.1 * dt

        model = create_model(
            config, models.RayleighTaylor2D,
            t_max=t1, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4,
        )

        temp_pred = []
        for k in range(windows):
            params = restore(config, model, ckpt_root, run, windowed=True, window=k + 1)
            temp_pred.append(
                predict_snapshots(model.temp_pred_fn, params, t_star, jnp.asarray(mesh))
            )
        temp_pred = np.concatenate(temp_pred)

    # mesh is y-major (x varies fastest): reshape to (T, ny, nx), then (T, nx, ny)
    mesh = np.asarray(mesh)
    nx = len(np.unique(np.round(mesh[:, 0], 6)))
    ny = len(np.unique(np.round(mesh[:, 1], 6)))
    ref = np.asarray(temp_ref)[: windows * num_time_steps]
    error = rel_l2(temp_pred, ref)

    ref_g = ref.reshape(-1, ny, nx).transpose(0, 2, 1)
    pred_g = temp_pred.reshape(-1, ny, nx).transpose(0, 2, 1)
    panels_2d(name, ref_g[-1], pred_g[-1], "RdYlBu_r", vsym=False)
    animate_2d(name, ref_g, pred_g, t_ref[: windows * num_time_steps],
               "RdYlBu_r", vsym=False)
    return {"temp_error": error}


def gen_bfs_flow(ckpt_root):
    name, run = "bfs_flow", run_for("bfs_flow")
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
    # The channel is 15:1 — keep the canvas as short as the strips, with
    # slim horizontal colorbars, so the figure has no dead whitespace.
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 3.4), dpi=150, constrained_layout=True)
    for ax, field, title, cmap in [
        (axes[0], U_ref, "Reference", "viridis"),
        (axes[1], U_pred, "PINN prediction", "viridis"),
        (axes[2], np.abs(U_pred - U_ref), "Absolute error", "magma"),
    ]:
        im = ax.tricontourf(coords[:, 0], coords[:, 1], field, levels=100, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.015, aspect=12)
        cbar.ax.tick_params(labelsize=6)
        cbar.locator = matplotlib.ticker.MaxNLocator(4)
        cbar.update_ticks()
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
    "rayleigh_taylor": gen_rayleigh_taylor,
    "bfs_flow": gen_bfs_flow,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--only", default=None, help="comma-separated example names")
    parser.add_argument(
        "--runs", default=None,
        help="per-example run overrides, e.g. gray_scott=gray_scott__pseudo_time,"
             "ks=ks__baseline (default: <example>__baseline)",
    )
    parser.add_argument(
        "--windows", default=None,
        help="per-example time-window count of the chosen run, e.g. "
             "gray_scott=4,kolmogorov_flow=4",
    )
    parser.add_argument(
        "--time-ranges", default=None,
        help="per-example time_range of the chosen run when it differs from "
             "the config default, e.g. gray_scott=0:1.0",
    )
    args = parser.parse_args()

    global RUN_OVERRIDES, WINDOW_OVERRIDES, TIME_RANGE_OVERRIDES
    if args.runs:
        RUN_OVERRIDES = dict(pair.split("=") for pair in args.runs.split(","))
    if args.windows:
        WINDOW_OVERRIDES = dict(pair.split("=") for pair in args.windows.split(","))
    if args.time_ranges:
        TIME_RANGE_OVERRIDES = {
            name: tuple(float(v) for v in rng.split(":"))
            for name, rng in (pair.split("=") for pair in args.time_ranges.split(","))
        }

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
