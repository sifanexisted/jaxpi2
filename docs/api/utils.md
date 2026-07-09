# jaxpi.utils

Small utilities: pytree flattening, update schedules, schedule-free eval params.

## `flatten_pytree()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/utils.py#L8' target='_blank'>[source]</a>

```python
flatten_pytree(pytree)
```

## `get_eval_params()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/utils.py#L24' target='_blank'>[source]</a>

```python
get_eval_params(state, schedule_free)
```

Parameters to use for inference.

With the schedule-free optimizer wrapper, the parameters held by the train
state are the training iterates; evaluation (error logging, IC propagation
between time windows, final prediction) should use the schedule-free
averaged parameters instead.

## `create_update_scheduler()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/utils.py#L40' target='_blank'>[source]</a>

```python
create_update_scheduler(every: int = 100, start: int = 0) -> Callable[[int], bool]
```

Build and return a step-checker for the given schedule.
