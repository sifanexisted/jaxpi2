# jaxpi.evaluator

Metric logging during training.

## `BaseEvaluator` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L8' target='_blank'>[source]</a>

### `BaseEvaluator.log_lr()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L13' target='_blank'>[source]</a>

```python
BaseEvaluator.log_lr(self, model, state)
```

### `BaseEvaluator.log_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L17' target='_blank'>[source]</a>

```python
BaseEvaluator.log_losses(self, loss_dict)
```

### `BaseEvaluator.log_raw_losses()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L21' target='_blank'>[source]</a>

```python
BaseEvaluator.log_raw_losses(self, model, params, state, batch)
```

### `BaseEvaluator.log_loss_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L28' target='_blank'>[source]</a>

```python
BaseEvaluator.log_loss_weights(self, state)
```

### `BaseEvaluator.log_pts_weights()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L33' target='_blank'>[source]</a>

```python
BaseEvaluator.log_pts_weights(self, state)
```

### `BaseEvaluator.log_grads()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L38' target='_blank'>[source]</a>

```python
BaseEvaluator.log_grads(self, model, params, batch)
```

### `BaseEvaluator.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/evaluator.py#L45' target='_blank'>[source]</a>

```python
BaseEvaluator.__call__(self, model, state, loss_dict, batch, *args)
```

Call self as a function.
