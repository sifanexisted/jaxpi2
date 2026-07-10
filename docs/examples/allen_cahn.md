# Allen–Cahn

Phase separation with a stiff double-well reaction term:

$$
\frac{\partial u}{\partial t} - 10^{-4}\,\frac{\partial^2 u}{\partial x^2} + 5u^3 - 5u = 0,
\qquad (t, x) \in [0, 1] \times [-1, 1],
$$

with periodic boundary conditions and $u(0, x) = x^2 \cos(\pi x)$. Sharp transition layers
form early and persist — a classic hard case for vanilla PINNs.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>4.0e-06</strong></span>
  <span>recipe <strong>baseline</strong> (PirateNet + SOAP + grad-norm balancing + causal)</span>
  <span>100k steps, single GPU</span>
</div>

The standard JAXPI recipe solves Allen–Cahn to machine-practical accuracy: the trained
network matches the reference to a relative L2 error of **4.0e-06**, with the absolute
error concentrated in thin bands around the transition layers and three orders of
magnitude below the solution scale. Historically this benchmark is where PINNs collapse to
the trivial equilibrium without causal weighting; interestingly, our experiments show the
modern stack is so robust here that disabling causal weighting barely matters (4.6e-06) —
the remaining ingredients compensate. Adaptive pseudo-time stepping is equally harmless
(4.1e-06) but unnecessary.

<figure class="example-figure">

![Allen-Cahn prediction vs reference](/jaxpi2/results/allen_cahn_pred.png)

<figcaption>Reference, PINN prediction, and absolute error over the full space-time domain.</figcaption>
</figure>

<figure class="example-figure">

![Allen-Cahn convergence](/jaxpi2/results/allen_cahn_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase run.</figcaption>
</figure>

## Run

```bash
cd examples/allen_cahn
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- The standard testbed for the causal loss — see the
  [causal training deep-dive](/methods/causal-training).
