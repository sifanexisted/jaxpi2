# jaxpi.models

PINN base classes, model/optimizer factories, and the sharded training step.

## `TrainState` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L35' target='_blank'>[source]</a>

*Bases: `TrainState`*

TrainState(step: int | jax.Array, apply_fn: collections.abc.Callable, params: flax.core.frozen_dict.FrozenDict[str, typing.Any], tx: optax._src.base.GradientTransformation, opt_state: Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, float, complex, Iterable[ForwardRef('ArrayTree')], Mapping[Any, ForwardRef('ArrayTree')]], loss_weights: Dict, pts_weights: Dict, momentum: float, pts_momentum: float, prev_params: Any = None)

### `TrainState.apply_loss_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L42' target='_blank'>[source]</a>

```python
TrainState.apply_loss_weights(self, loss_weights, **kwargs)
```

### `TrainState.apply_pts_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L54' target='_blank'>[source]</a>

```python
TrainState.apply_pts_weights(self, pts_weights, **kwargs)
```

### `TrainState.replace()`

```python
TrainState.replace(self, **updates)
```

Returns a new object replacing the specified fields with new values.

## `create_arch()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L68' target='_blank'>[source]</a>

```python
create_arch(config)
```

## `create_lr_schedule()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L86' target='_blank'>[source]</a>

```python
create_lr_schedule(config)
```

## `create_optimizer()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L109' target='_blank'>[source]</a>

```python
create_optimizer(config, lr)
```

## `create_model()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L149' target='_blank'>[source]</a>

```python
create_model(config, model_cls, *model_args, params=None, **model_kwargs)
```

Build a model from its config: lr schedule, optimizer, arch, state.

`model_args` / `model_kwargs` are forwarded to the model constructor
(problem-specific arguments such as initial conditions or physical
parameters). `params` warm-starts the train state (transfer learning).

## `create_train_state()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L163' target='_blank'>[source]</a>

```python
create_train_state(
    config,
    tx,
    arch,
    params=None,
    train_state_cls=<class 'jaxpi.models.TrainState'>,
)
```

## `PINN` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L186' target='_blank'>[source]</a>

### `PINN.neural_net()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L232' target='_blank'>[source]</a>

```python
PINN.neural_net(self, params, *args)
```

### `PINN.r_net()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L235' target='_blank'>[source]</a>

```python
PINN.r_net(self, params, *args)
```

### `PINN.sol_net()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L238' target='_blank'>[source]</a>

```python
PINN.sol_net(self, params, *args)
```

neural_net's outputs as a dict keyed by `variables`.

With residuals keyed by variable name, the pseudo-time damping pairs
each residual with its own solution component by key — immune to any
ordering (JAX sorts dict keys when flattening pytrees). No override
needed: declaring `variables` is sufficient.

### `PINN.losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L251' target='_blank'>[source]</a>

```python
PINN.losses(self, params, state, batch)
```

### `PINN.compute_pts_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L323' target='_blank'>[source]</a>

```python
PINN.compute_pts_weights(self, state, init_state, batch)
```

### `PINN.compute_loss_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L400' target='_blank'>[source]</a>

```python
PINN.compute_loss_weights(self, state, batch)
```

Balance losses based on the gradient norms of each loss.

### `PINN.loss()`

```python
PINN.loss(self, params, state, batch)
```

### `PINN.create_step_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L436' target='_blank'>[source]</a>

```python
PINN.create_step_fn(self)
```

### `PINN.create_update_loss_weights_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L469' target='_blank'>[source]</a>

```python
PINN.create_update_loss_weights_fn(self)
```

### `PINN.create_compute_raw_residual_losses_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L488' target='_blank'>[source]</a>

```python
PINN.create_compute_raw_residual_losses_fn(self)
```

Sharded unweighted residual losses (no pseudo-time, no causal) —
used by evaluators, so that full multi-GPU batches fit in memory.

### `PINN.create_compute_grad_norms_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L510' target='_blank'>[source]</a>

```python
PINN.create_compute_grad_norms_fn(self)
```

Sharded per-term gradient norms — used by evaluators.

### `PINN.create_update_pts_weights_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L528' target='_blank'>[source]</a>

```python
PINN.create_update_pts_weights_fn(self)
```

## `ForwardIVP` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L550' target='_blank'>[source]</a>

*Bases: `PINN`*

### `ForwardIVP.create_compute_causal_weights_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L601' target='_blank'>[source]</a>

```python
ForwardIVP.create_compute_causal_weights_fn(self)
```

### `ForwardIVP.compute_residual_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L661' target='_blank'>[source]</a>

```python
ForwardIVP.compute_residual_losses(
    self,
    params,
    state,
    batch,
    pseudo_time=False,
    causal=False,
)
```

## `ForwardBVP` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L680' target='_blank'>[source]</a>

*Bases: `PINN`*

### `ForwardBVP.compute_residual_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L684' target='_blank'>[source]</a>

```python
ForwardBVP.compute_residual_losses(self, params, state, batch, pseudo_time=False)
```
