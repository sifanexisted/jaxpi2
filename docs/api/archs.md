# jaxpi.archs

Flax network architectures and input embeddings.

## `PeriodEmbs` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/archs.py#L27' target='_blank'>[source]</a>

*Bases: `Module`*

Periodic cos/sin embeddings for selected input axes.

Note: despite its name, `period` is an angular frequency: axis i is
embedded as (cos(period_i * x), sin(period_i * x)), so the actual spatial
period is 2*pi / period_i. E.g. period=2*pi for a domain of length 1,
period=1.0 for a domain of length 2*pi.

### `PeriodEmbs.setup()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L42' target='_blank'>[source]</a>

```python
PeriodEmbs.setup(self)
```

Initializes a Module lazily (similar to a lazy ``__init__``).

``setup`` is called once lazily on a module instance when a module
is bound, immediately before any other methods like ``__call__`` are
invoked, or before a ``setup``-defined attribute on ``self`` is accessed.

This can happen in three cases:

  1. Immediately when invoking :meth:`apply`, :meth:`init` or
     :meth:`init_and_output`.

  2. Once the module is given a name by being assigned to an attribute of
     another module inside the other module's ``setup`` method
     (see :meth:`__setattr__`)::

        >>> class MyModule(nn.Module):
        ...   def setup(self):
        ...     submodule = nn.Conv(...)

        ...     # Accessing `submodule` attributes does not yet work here.

        ...     # The following line invokes `self.__setattr__`, which gives
        ...     # `submodule` the name "conv1".
        ...     self.conv1 = submodule

        ...     # Accessing `submodule` attributes or methods is now safe and
        ...     # either causes setup() to be called once.

  3. Once a module is constructed inside a method wrapped with
     :meth:`compact`, immediately before another method is called or
     ``setup`` defined attribute is accessed.

### `PeriodEmbs.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L55' target='_blank'>[source]</a>

```python
PeriodEmbs.__call__(self, x)
```

Apply the period embeddings to the specified axes.

## `FourierEmbs` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/archs.py#L72' target='_blank'>[source]</a>

*Bases: `Module`*

FourierEmbs(embed_scale: float, embed_dim: int, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7fa9d0b1edb0>, name: Optional[str] = None)

### `FourierEmbs.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L76' target='_blank'>[source]</a>

```python
FourierEmbs.__call__(self, x)
```

Call self as a function.

## `Mlp` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/archs.py#L90' target='_blank'>[source]</a>

*Bases: `Module`*

Mlp(arch_name: Optional[str] = 'Mlp', num_layers: int = 4, hidden_dim: int = 256, out_dim: int = 1, activation: str = 'tanh', periodicity: Optional[Dict] = None, fourier_emb: Optional[Dict] = None, nonlinearity: Union[int, list] = 0.0, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7fa9d0b1edb0>, name: Optional[str] = None)

### `Mlp.setup()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L100' target='_blank'>[source]</a>

```python
Mlp.setup(self)
```

Initializes a Module lazily (similar to a lazy ``__init__``).

``setup`` is called once lazily on a module instance when a module
is bound, immediately before any other methods like ``__call__`` are
invoked, or before a ``setup``-defined attribute on ``self`` is accessed.

This can happen in three cases:

  1. Immediately when invoking :meth:`apply`, :meth:`init` or
     :meth:`init_and_output`.

  2. Once the module is given a name by being assigned to an attribute of
     another module inside the other module's ``setup`` method
     (see :meth:`__setattr__`)::

        >>> class MyModule(nn.Module):
        ...   def setup(self):
        ...     submodule = nn.Conv(...)

        ...     # Accessing `submodule` attributes does not yet work here.

        ...     # The following line invokes `self.__setattr__`, which gives
        ...     # `submodule` the name "conv1".
        ...     self.conv1 = submodule

        ...     # Accessing `submodule` attributes or methods is now safe and
        ...     # either causes setup() to be called once.

  3. Once a module is constructed inside a method wrapped with
     :meth:`compact`, immediately before another method is called or
     ``setup`` defined attribute is accessed.

### `Mlp.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L103' target='_blank'>[source]</a>

```python
Mlp.__call__(self, x)
```

Call self as a function.

## `ModifiedMlp` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/archs.py#L120' target='_blank'>[source]</a>

*Bases: `Module`*

ModifiedMlp(arch_name: Optional[str] = 'ModifiedMlp', num_layers: int = 4, hidden_dim: int = 256, out_dim: int = 1, activation: str = 'tanh', periodicity: Optional[Dict] = None, fourier_emb: Optional[Dict] = None, nonlinearity: Union[int, list] = 0.0, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7fa9d0b1edb0>, name: Optional[str] = None)

### `ModifiedMlp.setup()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L130' target='_blank'>[source]</a>

```python
ModifiedMlp.setup(self)
```

Initializes a Module lazily (similar to a lazy ``__init__``).

``setup`` is called once lazily on a module instance when a module
is bound, immediately before any other methods like ``__call__`` are
invoked, or before a ``setup``-defined attribute on ``self`` is accessed.

This can happen in three cases:

  1. Immediately when invoking :meth:`apply`, :meth:`init` or
     :meth:`init_and_output`.

  2. Once the module is given a name by being assigned to an attribute of
     another module inside the other module's ``setup`` method
     (see :meth:`__setattr__`)::

        >>> class MyModule(nn.Module):
        ...   def setup(self):
        ...     submodule = nn.Conv(...)

        ...     # Accessing `submodule` attributes does not yet work here.

        ...     # The following line invokes `self.__setattr__`, which gives
        ...     # `submodule` the name "conv1".
        ...     self.conv1 = submodule

        ...     # Accessing `submodule` attributes or methods is now safe and
        ...     # either causes setup() to be called once.

  3. Once a module is constructed inside a method wrapped with
     :meth:`compact`, immediately before another method is called or
     ``setup`` defined attribute is accessed.

### `ModifiedMlp.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L133' target='_blank'>[source]</a>

```python
ModifiedMlp.__call__(self, x)
```

Call self as a function.

## `PirateBlock` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/archs.py#L157' target='_blank'>[source]</a>

*Bases: `Module`*

PirateBlock(hidden_dim: int, activation: str, nonlinearity: float, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7fa9d0b1edb0>, name: Optional[str] = None)

### `PirateBlock.setup()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L162' target='_blank'>[source]</a>

```python
PirateBlock.setup(self)
```

Initializes a Module lazily (similar to a lazy ``__init__``).

``setup`` is called once lazily on a module instance when a module
is bound, immediately before any other methods like ``__call__`` are
invoked, or before a ``setup``-defined attribute on ``self`` is accessed.

This can happen in three cases:

  1. Immediately when invoking :meth:`apply`, :meth:`init` or
     :meth:`init_and_output`.

  2. Once the module is given a name by being assigned to an attribute of
     another module inside the other module's ``setup`` method
     (see :meth:`__setattr__`)::

        >>> class MyModule(nn.Module):
        ...   def setup(self):
        ...     submodule = nn.Conv(...)

        ...     # Accessing `submodule` attributes does not yet work here.

        ...     # The following line invokes `self.__setattr__`, which gives
        ...     # `submodule` the name "conv1".
        ...     self.conv1 = submodule

        ...     # Accessing `submodule` attributes or methods is now safe and
        ...     # either causes setup() to be called once.

  3. Once a module is constructed inside a method wrapped with
     :meth:`compact`, immediately before another method is called or
     ``setup`` defined attribute is accessed.

### `PirateBlock.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L165' target='_blank'>[source]</a>

```python
PirateBlock.__call__(self, x, u, v)
```

Call self as a function.

## `PirateNet` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/archs.py#L188' target='_blank'>[source]</a>

*Bases: `Module`*

PirateNet(arch_name: Optional[str] = 'PirateNet', num_layers: int = 2, hidden_dim: int = 256, out_dim: int = 1, activation: str = 'tanh', nonlinearity: Union[int, list] = 0.0, periodicity: Optional[Dict] = None, fourier_emb: Optional[Dict] = None, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7fa9d0b1edb0>, name: Optional[str] = None)

### `PirateNet.setup()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L198' target='_blank'>[source]</a>

```python
PirateNet.setup(self)
```

Initializes a Module lazily (similar to a lazy ``__init__``).

``setup`` is called once lazily on a module instance when a module
is bound, immediately before any other methods like ``__call__`` are
invoked, or before a ``setup``-defined attribute on ``self`` is accessed.

This can happen in three cases:

  1. Immediately when invoking :meth:`apply`, :meth:`init` or
     :meth:`init_and_output`.

  2. Once the module is given a name by being assigned to an attribute of
     another module inside the other module's ``setup`` method
     (see :meth:`__setattr__`)::

        >>> class MyModule(nn.Module):
        ...   def setup(self):
        ...     submodule = nn.Conv(...)

        ...     # Accessing `submodule` attributes does not yet work here.

        ...     # The following line invokes `self.__setattr__`, which gives
        ...     # `submodule` the name "conv1".
        ...     self.conv1 = submodule

        ...     # Accessing `submodule` attributes or methods is now safe and
        ...     # either causes setup() to be called once.

  3. Once a module is constructed inside a method wrapped with
     :meth:`compact`, immediately before another method is called or
     ``setup`` defined attribute is accessed.

### `PirateNet.__call__()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/.venv/lib/python3.12/site-packages/flax/linen/module.py#L207' target='_blank'>[source]</a>

```python
PirateNet.__call__(self, x)
```

Call self as a function.
