"""Generate gallery figures and animations for the docs website.

Renders solution fields directly from each example's reference data (no
training required) into docs/public/gallery/. Run from the repo root:

    python docs/scripts/gen_gallery.py

The Kolmogorov-flow Re=1e6 assets are only rendered when its dataset is
available (pass --kf1e6-data or place the file under the example's data/).
"""

import argparse
import os
import sys
from contextlib import contextmanager

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "public", "gallery")


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
        sys.modules.pop("utils", None)


def load_dataset(name, *args, **kwargs):
    with example(name):
        import utils

        return utils.get_dataset(*args, **kwargs)


def save_field(name, field, cmap, vsym=False, figsize=(4.8, 4.0)):
    """Full-bleed rendering of a 2D field (no axes) for gallery cards."""
    field = np.asarray(field)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    kwargs = {}
    if vsym:
        vmax = np.percentile(np.abs(field), 99.5)
        kwargs = {"vmin": -vmax, "vmax": vmax}
    ax.imshow(field.T, origin="lower", cmap=cmap, aspect="auto", **kwargs)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {os.path.relpath(path, REPO)}")


def save_animation(name, frames, cmap, fps=12, vsym=True, size=480):
    """Encode a (t, nx, ny) sequence as an mp4 loop."""
    frames = np.asarray(frames)
    if vsym:
        vmax = np.percentile(np.abs(frames), 99.5)
        vmin = -vmax
    else:
        vmin, vmax = np.percentile(frames, [0.5, 99.5])

    colormap = plt.get_cmap(cmap)
    path = os.path.join(OUT, f"{name}.mp4")
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=7,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        norm = np.clip((frame.T[::-1] - vmin) / (vmax - vmin), 0, 1)
        rgb = (colormap(norm)[..., :3] * 255).astype(np.uint8)
        # resize to a fixed even size with simple striding/repeat
        h, w = rgb.shape[:2]
        scale = max(1, int(round(max(h, w) / size)))
        rgb = rgb[::scale, ::scale]
        rgb = rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2]
        writer.append_data(rgb)
    writer.close()
    print(f"  {os.path.relpath(path, REPO)} ({os.path.getsize(path) // 1024} KB)")


def spacetime(name, u, cmap, vsym=True):
    """1D problems: space-time diagram u(t, x) with x vertical."""
    save_field(name, np.asarray(u), cmap, vsym=vsym)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kf1e6-data",
        default=os.path.join(REPO, "examples/kolmogorov_flow_Re1e6/data/kolmogorov_flow_Re1e6.npy"),
    )
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------------------- 1D problems
    print("advection")
    u, t, x = load_dataset("advection")
    spacetime("advection", u, "twilight_shifted")

    print("wave")
    u, t, x = load_dataset("wave")
    spacetime("wave", u, "RdBu_r")

    print("allen_cahn")
    u, t, x = load_dataset("allen_cahn")
    spacetime("allen_cahn", u, "Spectral_r")

    print("burgers")
    u, t, x = load_dataset("burgers")
    spacetime("burgers", u, "coolwarm")

    print("inviscid_burgers")
    u, t, x = load_dataset("inviscid_burgers")
    spacetime("inviscid_burgers", u, "coolwarm")

    print("kdv")
    u, t, x = load_dataset("kdv")
    spacetime("kdv", u, "turbo", vsym=False)

    print("ks")
    u, t, x = load_dataset("ks", time_range=[0.0, 1.0])
    spacetime("ks", u, "inferno", vsym=False)

    print("sod_shock_tube")
    rho, u, p, *_ = load_dataset("sod_shock_tube")
    save_field("sod_shock_tube", np.asarray(rho).T, "cividis", vsym=False)

    # ------------------------------------------------------------- 2D problems
    print("ginzburg_landau")
    u, v, t, x, y, *_ = load_dataset("ginzburg_landau", time_range=[0.0, 1.0])
    u = np.asarray(u)  # (t, nx, ny)
    save_field("ginzburg_landau", u[-1], "twilight")
    save_animation("ginzburg_landau", u[:: max(1, len(u) // 90)], "twilight")

    print("gray_scott")
    u, v, t, x, y, *_ = load_dataset("gray_scott", time_range=[0.0, 1.0])
    v = np.asarray(v)
    save_field("gray_scott", v[-1], "magma", vsym=False)
    save_animation("gray_scott", v[:: max(1, len(v) // 90)], "magma", vsym=False)

    print("kolmogorov_flow")
    u, v, w, t, coords, nu = load_dataset("kolmogorov_flow", time_range=[0.0, 1.0])
    n = int(round(np.sqrt(w.shape[1])))
    w = np.asarray(w).reshape(-1, n, n)
    save_field("kolmogorov_flow", w[-1], "RdBu_r")
    save_animation("kolmogorov_flow", w[:: max(1, len(w) // 90)], "RdBu_r", fps=10)
    # hero animation for the landing page
    save_animation("hero", w[:: max(1, len(w) // 120)], "RdBu_r", fps=12, size=640)

    print("rayleigh_taylor")
    uv, p, temp, t, mesh, *_ = load_dataset("rayleigh_taylor", time_range=[0.1, 1.0])
    mesh = np.asarray(mesh)
    nx = len(np.unique(np.round(mesh[:, 0], 6)))
    ny = len(np.unique(np.round(mesh[:, 1], 6)))
    # The mesh is y-major (x varies fastest): frame layout is (ny, nx).
    # Transpose to (x, y) so save_field/save_animation display y upward.
    temp = np.asarray(temp).reshape(-1, ny, nx).transpose(0, 2, 1)
    save_field("rayleigh_taylor", temp[-1], "RdYlBu_r", vsym=False, figsize=(2.4, 4.8))
    save_animation("rayleigh_taylor", temp[:: max(1, len(temp) // 90)], "RdYlBu_r", vsym=False)

    # ---------------------------------------------------- boundary-value flows
    print("lid_driven_cavity")
    u, v, x, y, nu = load_dataset("lid_driven_cavity", 5000)
    speed = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2)
    save_field("lid_driven_cavity", speed, "viridis", vsym=False)

    print("bfs_flow")
    u, v, p, coords, *_ = load_dataset("bfs_flow")
    speed = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2)
    coords = np.asarray(coords)
    fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=170)
    ax.tricontourf(coords[:, 0], coords[:, 1], speed, levels=120, cmap="viridis")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, "bfs_flow.png"))
    plt.close(fig)
    print(f"  docs/public/gallery/bfs_flow.png")

    # ------------------------------------------------------------- 3D problems
    print("taylor_green")
    # Analytic IC u = (sin x cos y cos z, -cos x sin y cos z, 0). Render the
    # periodic cube with the vorticity component normal to each visible face:
    # w_x = -cos x sin y sin z, w_y = -sin x cos y sin z, w_z = 2 sin x sin y cos z.
    n = 200
    xs = np.linspace(0, 2 * np.pi, n)
    A, B = np.meshgrid(xs, xs, indexing="ij")
    cmap = plt.get_cmap("RdBu_r")
    norm = lambda f, s: cmap((f / s + 1.0) / 2.0)
    L = 2 * np.pi

    fig = plt.figure(figsize=(4.8, 4.4), dpi=160)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    surf_kw = dict(rstride=1, cstride=1, shade=False, antialiased=False)
    # top face z = L: w_z(A, B, L) = 2 sin A sin B
    ax.plot_surface(A, B, np.full_like(A, L),
                    facecolors=norm(2.0 * np.sin(A) * np.sin(B), 2.0), **surf_kw)
    # front face y = 0: w_y(A, 0, B) = -sin A sin B
    ax.plot_surface(A, np.zeros_like(A), B,
                    facecolors=norm(-np.sin(A) * np.sin(B), 1.0), **surf_kw)
    # side face x = L: w_x(L, A, B) = -sin A sin B
    ax.plot_surface(np.full_like(A, L), A, B,
                    facecolors=norm(-np.sin(A) * np.sin(B), 1.0), **surf_kw)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, L), ax.set_ylim(0, L), ax.set_zlim(0, L)
    fig.subplots_adjust(-0.18, -0.18, 1.18, 1.18)
    fig.savefig(os.path.join(OUT, "taylor_green.png"), transparent=True)
    plt.close(fig)
    print("  docs/public/gallery/taylor_green.png")

    print("kolmogorov_flow_Re1e6")
    if os.path.exists(args.kf1e6_data):
        data = np.load(args.kf1e6_data, allow_pickle=True).item()
        w = np.asarray(data["vorticity"])
        n = int(round(np.sqrt(w.shape[1])))
        w = w.reshape(-1, n, n)[:, ::2, ::2]  # 2048 -> 1024
        save_field("kolmogorov_flow_Re1e6", w[0], "RdBu_r")
        save_animation(
            "kolmogorov_flow_Re1e6", w[:: max(1, len(w) // 60), ::2, ::2], "RdBu_r", fps=8
        )
    else:
        print("  dataset not found, skipped")

    total = sum(
        os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)
    )
    print(f"\nTotal gallery size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
