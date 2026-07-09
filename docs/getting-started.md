# Getting Started

## Installation

JAXPI requires Python 3.11+. The pinned stack (JAX 0.10, Flax 0.12) targets Linux with a local
CUDA installation; everything also runs on CPU.

```bash
git clone https://github.com/sifanexisted/jaxpi2.git
cd jaxpi2
pip install -e .
```

To run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Train your first PINN

Every example is a self-contained directory under `examples/`. Training metrics are logged to
[Weights & Biases](https://wandb.ai) — log in first (`wandb login`) or run offline with
`WANDB_MODE=offline`.

```bash
cd examples/advection
python3 main.py --config=configs/baseline.py
```

You will see a periodic table of losses and errors in the console:

```
[12:57:54 - main - INFO]  Iter: 500    Time: 12.3
[12:57:54 - main - INFO]  ---------  ----------
[12:57:54 - main - INFO]  ics_loss   3.300e-01
[12:57:54 - main - INFO]  res_loss   1.022e-01
[12:57:54 - main - INFO]  l2_error   8.410e-02
```

## Configuration

Hyperparameters live in `configs/` as [`ml_collections`](https://github.com/google/ml_collections)
configs. `base.py` holds the full tree; variants like `baseline.py` or `pseudo_time.py` are thin
overrides. Any field can also be overridden from the command line:

```bash
python3 main.py --config=configs/baseline.py \
    --config.training.batch_size=2048 \
    --config.optim.learning_rate=5e-4 \
    --config.arch.arch_name=modifiedmlp
```

The main config sections:

| Section | Controls |
| --- | --- |
| `arch` | Network architecture, width/depth, embeddings |
| `optim` | Optimizer (Adam / SOAP / Muon), LR schedule, schedule-free wrapper |
| `training` | Steps, batch size, time windows, transfer learning, resume |
| `loss_weighting` | Static weights or grad-norm balancing schedule |
| `pseudo_time` | Pseudo-time stepping (see [Training Techniques](/guide/training-techniques)) |
| `causal` | Causal weighting tolerance and chunking |
| `logging` / `saving` | W&B logging cadence, checkpoint cadence |

## Multi-GPU training

Training is data-parallel out of the box — batches are sharded across all visible devices with
`jax.shard_map`, and gradients are averaged exactly, so results match single-GPU runs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config=configs/baseline.py
```

The batch size must be divisible by the number of devices (you get a clear error otherwise).

::: tip Out of memory?
Reduce `--config.training.batch_size` — it is the dominant memory knob.
:::

## Resuming a run

All examples resume from their latest checkpoint:

```bash
python3 main.py --config=configs/baseline.py --config.training.resume=True
```

Single-run examples continue from the last saved step; time-windowed examples continue from the
last trained window; the multi-stage Taylor–Green cascade continues mid-stage.

## Evaluation

Each example ships an `eval.ipynb` that restores the latest checkpoint and compares against the
reference solution:

```bash
cd examples/advection
jupyter lab eval.ipynb
```

## Next steps

- [Core Concepts](/guide/concepts) — how models, samplers, and the trainer fit together
- [Write Your Own Example](/guide/new-example) — solve your own PDE end to end
- [Evaluation](/guide/evaluation) — restore checkpoints and compute errors
- [Examples Gallery](/examples/) — all benchmarks with figures and run commands
- [Methods](/methods/piratenet) — illustrated deep dives into the baseline's components
