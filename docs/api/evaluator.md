# jaxpi.evaluator

Metric logging.

Metric names use "section/name" keys (e.g. "loss/res", "error/u",
"weights/u_ic") so that W&B groups them into separate chart sections:
losses, errors, adaptive weights, gradient norms, etc.

## `BaseEvaluator` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L9' target='_blank'>[source]</a>

### `BaseEvaluator.log_lr()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L14' target='_blank'>[source]</a>

```python
BaseEvaluator.log_lr(self, model, state)
```

### `BaseEvaluator.log_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L18' target='_blank'>[source]</a>

```python
BaseEvaluator.log_losses(self, loss_dict)
```

### `BaseEvaluator.log_raw_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L22' target='_blank'>[source]</a>

```python
BaseEvaluator.log_raw_losses(self, model, params, state, batch)
```

### `BaseEvaluator.log_loss_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L30' target='_blank'>[source]</a>

```python
BaseEvaluator.log_loss_weights(self, state)
```

### `BaseEvaluator.log_pts_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L35' target='_blank'>[source]</a>

```python
BaseEvaluator.log_pts_weights(self, state)
```

### `BaseEvaluator.log_grads()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L40' target='_blank'>[source]</a>

```python
BaseEvaluator.log_grads(self, model, state, batch)
```

### `BaseEvaluator.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L46' target='_blank'>[source]</a>

```python
BaseEvaluator.__call__(self, model, state, loss_dict, batch, *args)
```

Call self as a function.
