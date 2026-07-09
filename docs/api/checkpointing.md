# jaxpi.checkpointing

Orbax checkpointing and resume helpers.

## `get_ckpt_path()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L10' target='_blank'>[source]</a>

```python
get_ckpt_path(config)
```

Checkpoint root for a run: <saving.ckpt_path or cwd>/<wandb.name>/ckpt.

## `has_checkpoint_steps()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L17' target='_blank'>[source]</a>

```python
has_checkpoint_steps(path)
```

Whether an Orbax checkpoint directory contains at least one saved step.

## `latest_time_window()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L24' target='_blank'>[source]</a>

```python
latest_time_window(ckpt_path, pattern='time_window_(\\d+)')
```

Index of the last trained time window, based on checkpoint directories.

## `create_checkpoint_manager()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L36' target='_blank'>[source]</a>

```python
create_checkpoint_manager(config, ckpt_path, suffix=None)
```

## `save_checkpoint()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L49' target='_blank'>[source]</a>

```python
save_checkpoint(ckpt_mngr, state)
```

## `restore_checkpoint()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L57' target='_blank'>[source]</a>

```python
restore_checkpoint(ckpt_mngr, state, step=None)
```

## `CustomJSONEncoder` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L79' target='_blank'>[source]</a>

*Bases: `JSONEncoder`*

Extensible JSON <https://json.org> encoder for Python data structures.

Supports the following objects and types by default:

+-------------------+---------------+
| Python            | JSON          |
+===================+===============+
| dict              | object        |
+-------------------+---------------+
| list, tuple       | array         |
+-------------------+---------------+
| str               | string        |
+-------------------+---------------+
| int, float        | number        |
+-------------------+---------------+
| True              | true          |
+-------------------+---------------+
| False             | false         |
+-------------------+---------------+
| None              | null          |
+-------------------+---------------+

To extend this to recognize other objects, subclass and implement a
``.default()`` method with another method that returns a serializable
object for ``o`` if possible, otherwise it should call the superclass
implementation (to raise ``TypeError``).

### `CustomJSONEncoder.default()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L80' target='_blank'>[source]</a>

```python
CustomJSONEncoder.default(self, obj)
```

Implement this method in a subclass such that it returns
a serializable object for ``o``, or calls the base implementation
(to raise a ``TypeError``).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return super().default(o)

## `save_config()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/checkpointing.py#L88' target='_blank'>[source]</a>

```python
save_config(config, workdir, name=None)
```
