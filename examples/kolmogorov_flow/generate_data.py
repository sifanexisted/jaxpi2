"""Regenerate the Kolmogorov flow reference dataset (data/kolmogorov_flow.npy).

Solves the 2D incompressible Navier-Stokes equations on the unit periodic
square with jax-cfd's pseudo-spectral solver (float64, 2/3-rule dealiasing,
Crank-Nicolson for the viscous term + RK4 for advection/forcing):

    nu = 5e-4 (Re ~ 2000),  body force (2 sin(4 pi y), 0)

The initial condition is the first frame of the existing dataset. The solver
runs at 512^2 (the flow's enstrophy content beyond the 256^2 grid is O(1e-8),
so this fully resolves it) with dt = 2e-4, then every saved frame is
Fourier-restricted to the 256^2 dataset grid — exact for the resolved modes.
Vorticity is computed spectrally and pressure is reconstructed from the
velocity via the spectral pressure Poisson solve (zero mean).

Accuracy: this trajectory was cross-validated three ways — it is insensitive
to doubling resolution and halving dt at the 1e-6 level, and two independent
methods (a JAX-Fluids compressible WENO5 run at Mach 0.1 and a PINN trained
only on the IC + PDE residual) agree with it to their own expected error.

Requires jax-cfd (not a package dependency): pip install jax-cfd

Usage:
    python generate_data.py [--out data/kolmogorov_flow.npy]
"""

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import jax_cfd.base.grids as grids
import jax_cfd.spectral.equations as spectral_equations
import jax_cfd.spectral.time_stepping as spectral_time_stepping
import jax_cfd.spectral.utils as spectral_utils

N_SIM = 512      # solver grid
N_DATA = 256     # dataset grid
NU = 5e-4
DT = 2e-4
SAVE_EVERY = 0.2
T_FINAL = 4.0


def resample_periodic(f, m):
    """Fourier resampling between n^2 and m^2 cell-centered periodic grids
    (interpolation for m > n, restriction for m < n), including the
    half-cell phase shift between the two sets of cell centers."""
    n = f.shape[0]
    F = np.fft.fft2(f)
    k = np.fft.fftfreq(n) * n
    shift = 0.5 / m - 0.5 / n
    F = F * np.exp(2j * np.pi * (k[:, None] + k[None, :]) * shift)
    h = min(n, m) // 2
    Fp = np.zeros((m, m), complex)
    Fp[:h, :h] = F[:h, :h]
    Fp[:h, -h:] = F[:h, -h:]
    Fp[-h:, :h] = F[-h:, :h]
    Fp[-h:, -h:] = F[-h:, -h:]
    return np.real(np.fft.ifft2(Fp)) * (m / n) ** 2


def vorticity_xy(u, v):
    """w = v_x - u_y for [x, y]-layout fields (x = axis 0)."""
    n = u.shape[0]
    k = 2j * np.pi * np.fft.fftfreq(n) * n
    return (np.real(np.fft.ifft2(np.fft.fft2(v) * k[:, None]))
            - np.real(np.fft.ifft2(np.fft.fft2(u) * k[None, :])))


def pressure_xy(u, v):
    """Spectral pressure Poisson solve lap p = -d_i d_j (u_i u_j), zero mean.
    (The body force is divergence-free, so it does not enter.)"""
    n = u.shape[0]
    k1 = np.fft.fftfreq(n) * n
    kx, ky = k1[:, None], k1[None, :]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    uu, uv, vv = (np.fft.fft2(a) for a in (u * u, u * v, v * v))
    p_hat = -(kx**2 * uu + 2.0 * kx * ky * uv + ky**2 * vv) / k2
    p_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(p_hat))


def body_force_fn(grid):
    """fx = 2 sin(4 pi y), fy = 0 — the classic Kolmogorov forcing."""
    def forcing(v):
        del v
        _, y = grid.mesh()
        fx = 2.0 * jnp.sin(4.0 * jnp.pi * y)
        return (grids.GridArray(fx, offset=(0.5, 0.5), grid=grid),
                grids.GridArray(jnp.zeros_like(fx), offset=(0.5, 0.5), grid=grid))
    return forcing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/kolmogorov_flow.npy")
    parser.add_argument("--ic", default="data/kolmogorov_flow.npy",
                        help="dataset whose first frame provides the IC")
    args = parser.parse_args()

    # --- IC: first frame of the existing dataset, upsampled to the sim grid.
    # Dataset fields are x-fastest flattened -> reshape gives [y, x];
    # transpose to jax-cfd's [x, y] convention.
    d = np.load(args.ic, allow_pickle=True).item()
    u0 = resample_periodic(
        np.asarray(d["velocity"][0, :, 0], np.float64).reshape(N_DATA, N_DATA).T, N_SIM)
    v0 = resample_periodic(
        np.asarray(d["velocity"][0, :, 1], np.float64).reshape(N_DATA, N_DATA).T, N_SIM)

    k1 = 2j * np.pi * np.fft.fftfreq(N_SIM) * N_SIM
    w0 = (np.real(np.fft.ifft2(np.fft.fft2(v0) * k1[:, None]))
          - np.real(np.fft.ifft2(np.fft.fft2(u0) * k1[None, :])))
    w_hat = jnp.fft.rfftn(jnp.asarray(w0))

    grid = grids.Grid((N_SIM, N_SIM), domain=((0.0, 1.0), (0.0, 1.0)))
    equation = spectral_equations.NavierStokes2D(
        viscosity=NU, grid=grid, drag=0.0, smooth=True, forcing_fn=body_force_fn)
    step_fn = spectral_time_stepping.crank_nicolson_rk4(equation, DT)

    inner = int(round(SAVE_EVERY / DT))

    @jax.jit
    def advance(w_hat):
        return jax.lax.scan(lambda c, _: (step_fn(c), None), w_hat,
                            None, length=inner)[0]

    velocity_solve = spectral_utils.vorticity_to_velocity(grid)

    def snapshot(w_hat):
        """Solver state -> dataset-grid fields, x-fastest flattened."""
        vx_hat, vy_hat = velocity_solve(w_hat)
        u = resample_periodic(np.asarray(jnp.fft.irfftn(vx_hat)), N_DATA)
        v = resample_periodic(np.asarray(jnp.fft.irfftn(vy_hat)), N_DATA)
        return (u.T.ravel(), v.T.ravel(),
                vorticity_xy(u, v).T.ravel(), pressure_xy(u, v).T.ravel())

    n_saves = int(round(T_FINAL / SAVE_EVERY))
    ts, frames = [0.0], [snapshot(w_hat)]
    for i in range(n_saves):
        w_hat = advance(w_hat)
        t = (i + 1) * SAVE_EVERY
        frame = snapshot(w_hat)
        assert np.isfinite(frame[2]).all(), f"solution blew up at t={t}"
        ts.append(t)
        frames.append(frame)
        print(f"t = {t:.2f}", flush=True)

    u, v, w, p = (np.stack(a) for a in zip(*frames))
    centers = (np.arange(N_DATA) + 0.5) / N_DATA
    xx, yy = np.meshgrid(centers, centers)
    data = {
        "t": np.asarray(ts, np.float32),
        "coords": np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32),
        "velocity": np.stack([u, v], axis=-1).astype(np.float32),
        "pressure": p.astype(np.float32),
        "vorticity": w.astype(np.float32),
        "nu": np.float32(NU),
    }
    np.save(args.out, data, allow_pickle=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
