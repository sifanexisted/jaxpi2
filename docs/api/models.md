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

### `TrainState.replace()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/struct.py#L140' target='_blank'>[source]</a>

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

### `PINN.neural_net()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L220' target='_blank'>[source]</a>

```python
PINN.neural_net(self, params, *args)
```

### `PINN.r_net()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L223' target='_blank'>[source]</a>

```python
PINN.r_net(self, params, *args)
```

### `PINN.losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L226' target='_blank'>[source]</a>

```python
PINN.losses(self, params, state, batch)
```

### `PINN.compute_pts_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L264' target='_blank'>[source]</a>

```python
PINN.compute_pts_weights(self, state, init_state, batch)
```

### `PINN.compute_loss_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L331' target='_blank'>[source]</a>

```python
PINN.compute_loss_weights(self, state, batch)
```

Balance losses based on the gradient norms of each loss.

### `PINN.loss()`

```python
PINN.loss(self, params, state, batch)
```

### `PINN.create_step_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L378' target='_blank'>[source]</a>

```python
PINN.create_step_fn(self)
```

### `PINN.create_update_loss_weights_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L411' target='_blank'>[source]</a>

```python
PINN.create_update_loss_weights_fn(self)
```

### `PINN.create_update_pts_weights_fn()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L430' target='_blank'>[source]</a>

```python
PINN.create_update_pts_weights_fn(self)
```

## `ForwardIVP` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L452' target='_blank'>[source]</a>

*Bases: `PINN`*

### `ForwardIVP.compute_causal_weights()`

```python
ForwardIVP.compute_causal_weights(self, state, batch)
```

### `ForwardIVP.compute_residual_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L527' target='_blank'>[source]</a>

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

## `ForwardBVP` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L546' target='_blank'>[source]</a>

*Bases: `PINN`*

### `ForwardBVP.compute_residual_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/models.py#L550' target='_blank'>[source]</a>

```python
ForwardBVP.compute_residual_losses(self, params, state, batch, pseudo_time=False)
```
