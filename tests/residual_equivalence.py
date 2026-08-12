"""Residual equivalence harness for the forward-mode derivative migration.

Captures each example's vmapped r_net outputs on fixed random points with
deterministic (seeded) init parameters, then verifies the migrated code
reproduces them to float64 tolerance:

    python tests/residual_equivalence.py --capture   # on pristine code
    python tests/residual_equivalence.py --verify    # after migration

Models are built with synthetic constants and arrays: fixtures only prove
old code == new code on identical models, so the values need to be fixed,
not physical. Run with PYTHONPATH pointing at this repo root.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures"

import jaxpi

assert Path(jaxpi.__file__).resolve().is_relative_to(ROOT), (
    f"jaxpi resolves to {jaxpi.__file__}, not the worktree under {ROOT}; "
    "set PYTHONPATH to the worktree root"
)

from jaxpi.models import create_model  # noqa: E402


def load_example(name):
    """Import an example's models module and baseline config, isolated."""
    exdir = ROOT / "examples" / name.replace("_multistage", "")
    for key in list(sys.modules):
        if key == "models" or key == "utils" or key.startswith("configs"):
            del sys.modules[key]
    sys.path.insert(0, str(exdir))
    importlib.invalidate_caches()
    try:
        models = importlib.import_module("models")
        config = importlib.import_module("configs.baseline").get_config()
    finally:
        sys.path.remove(str(exdir))
    return models, config


def _arr(n=8):
    return jnp.linspace(0.1, 0.9, n)


def build(name):
    """Return (model, n_coords) for an example, synthetic constants."""
    models, config = load_example(name)
    kw1d = dict(u0=jnp.zeros(8), t_star=_arr(), x_star=_arr())

    if name == "advection":
        return create_model(config, models.Advection1D, c=1.5, **kw1d), 2
    if name == "allen_cahn":
        return create_model(config, models.AllenCahn, **kw1d), 2
    if name == "burgers":
        return create_model(config, models.Burgers, **kw1d), 2
    if name == "inviscid_burgers":
        return create_model(config, models.InviscidBurgers, **kw1d), 2
    if name == "kdv":
        return create_model(config, models.KDV, **kw1d), 2
    if name == "ks":
        return create_model(config, models.KS, **kw1d), 2
    if name == "wave":
        return create_model(config, models.Wave1D, c=2.0, **kw1d), 2
    if name == "ginzburg_landau":
        return create_model(config, models.GinzburgLandau,
                            t_max=1.0, eps=0.05, k=1.2), 3
    if name == "gray_scott":
        return create_model(config, models.GrayScott, t_max=1.0,
                            b1=0.04, b2=0.1, c1=1.0, c2=1.0,
                            eps1=2e-5, eps2=1e-5), 3
    if name == "kolmogorov_flow":
        return create_model(config, models.NavierStokes2D,
                            t_max=1.0, nu=1e-3), 3
    if name == "kolmogorov_flow_Re1e6":
        return create_model(config, models.NavierStokes2D,
                            t_max=0.2, nu=1e-6), 3
    if name == "lid_driven_cavity":
        return create_model(config, models.NavierStokes2D, nu=0.01), 2
    if name == "bfs_flow":
        pts = jnp.stack([_arr(4), _arr(4)], axis=1)
        return create_model(config, models.NavierStokes2D,
                            u_inflow=jnp.ones(4), inflow_coords=pts,
                            outflow_coords=pts, wall_coords=pts, nu=0.01), 2
    if name == "rayleigh_taylor":
        return create_model(config, models.RayleighTaylor2D, t_max=1.0,
                            alpha1=0.005, alpha2=1.0, alpha3=0.5,
                            alpha4=0.002), 3
    if name == "taylor_green":
        return create_model(config, models.NavierStokes3D,
                            t_max=1.0, nu=0.01), 4
    if name == "taylor_green_multistage":
        base = create_model(config, models.NavierStokes3D,
                            t_max=1.0, nu=0.01)
        return create_model(config, models.MultiStage, t_max=1.0, nu=0.01,
                            prev_params_list=[base.state.params],
                            eps_list=[1.0, 0.5]), 4
    if name == "sod_shock_tube":
        bc = jnp.stack([_arr(4), jnp.zeros(4)], axis=1)
        return create_model(config, models.Euler1D,
                            rho0=jnp.ones(8), u0=jnp.zeros(8),
                            p0=jnp.ones(8), t_star=_arr(), x_star=_arr(),
                            left_coords=bc, right_coords=bc), 2
    raise KeyError(name)


EXAMPLES = [
    "advection", "allen_cahn", "bfs_flow", "burgers", "ginzburg_landau",
    "gray_scott", "inviscid_burgers", "kdv", "kolmogorov_flow",
    "kolmogorov_flow_Re1e6", "ks", "lid_driven_cavity", "rayleigh_taylor",
    "sod_shock_tube", "taylor_green", "taylor_green_multistage", "wave",
]

N_POINTS = 192


def residuals(name):
    model, n_coords = build(name)
    rng = np.random.default_rng(42)
    coords = [jnp.asarray(rng.uniform(0.1, 0.9, N_POINTS)) for _ in range(n_coords)]
    out = jax.vmap(model.r_net, (None,) + (0,) * n_coords)(
        model.state.params, *coords)
    if isinstance(out, dict):
        return {k: np.asarray(v, np.float64) for k, v in out.items()}
    if isinstance(out, (tuple, list)):
        return {f"r{i}": np.asarray(v, np.float64) for i, v in enumerate(out)}
    return {"r": np.asarray(out, np.float64)}


def bench(name, n_pts=8192, reps=15):
    import time
    model, n_coords = build(name)
    rng = np.random.default_rng(0)
    coords = [jnp.asarray(rng.uniform(0.1, 0.9, n_pts))
              for _ in range(n_coords)]
    params = model.state.params
    f = jax.jit(jax.vmap(model.r_net, (None,) + (0,) * n_coords))
    jax.block_until_ready(f(params, *coords))
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(params, *coords))
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--bench", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()

    if args.bench:
        for name in (args.only or EXAMPLES):
            print(f"[bench] {name}: {bench(name):.2f} ms")
        return

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    names = args.only or EXAMPLES
    failures = []
    for name in names:
        res = residuals(name)
        path = FIXTURE_DIR / f"{name}.npz"
        if args.capture:
            np.savez(path, **res)
            print(f"[captured] {name}: {list(res)} ({N_POINTS} pts)")
            continue
        ref = np.load(path)
        worst = 0.0
        for k in ref.files:
            scale = np.max(np.abs(ref[k])) + 1e-30
            diff = np.max(np.abs(res[k] - ref[k])) / scale
            worst = max(worst, diff)
        status = "OK " if worst < 1e-8 else "FAIL"
        print(f"[{status}] {name}: max rel diff {worst:.2e}")
        if worst >= 1e-8:
            failures.append(name)
    if args.verify and failures:
        sys.exit(f"equivalence failures: {failures}")


if __name__ == "__main__":
    main()
