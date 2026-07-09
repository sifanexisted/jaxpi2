# Backward-facing Step

Steady channel flow over a backward-facing step at $\mathrm{Re} = 800$ — a boundary-value
problem with separation and reattachment:

$$
(\mathbf{u} \cdot \nabla)\,\mathbf{u} + \nabla p - \nu \nabla^2 \mathbf{u} = 0,
\qquad \nabla \cdot \mathbf{u} = 0,
$$

with a parabolic inflow profile $u(y) = 24\,y\,(0.5 - y)$, no-slip walls, and an outflow
condition at the channel exit.

<figure class="example-figure">

![Backward-facing step velocity magnitude](/jaxpi2/gallery/bfs_flow.png)

<figcaption>Reference velocity magnitude: recirculation bubble behind the step.</figcaption>
</figure>

## Run

```bash
cd examples/bfs_flow
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet baseline |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Collocation points are drawn from the unstructured reference mesh with `MeshSampler`
  rather than uniformly — the geometry is non-rectangular.
- Reading the `.vtu` reference data requires `pip install pyvista`.
