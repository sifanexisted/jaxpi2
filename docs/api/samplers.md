# jaxpi.samplers

Infinite collocation-point samplers.

## `BaseSampler` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/samplers.py#L9' target='_blank'>[source]</a>

Infinite iterator over randomly generated batches.

The global batch is sharded across devices by the training step, so the
batch size must be divisible by the number of devices.

### `BaseSampler.data_generation()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/samplers.py#L46' target='_blank'>[source]</a>

```python
BaseSampler.data_generation(self, key)
```

## `UniformSampler` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/samplers.py#L50' target='_blank'>[source]</a>

*Bases: `BaseSampler`*

Infinite iterator over randomly generated batches.

The global batch is sharded across devices by the training step, so the
batch size must be divisible by the number of devices.

### `UniformSampler.data_generation()`

```python
UniformSampler.data_generation(self, key)
```

Generates data containing batch_size samples

## `MeshSampler` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/samplers.py#L72' target='_blank'>[source]</a>

*Bases: `BaseSampler`*

Infinite iterator over randomly generated batches.

The global batch is sharded across devices by the training step, so the
batch size must be divisible by the number of devices.

### `MeshSampler.data_generation()`

```python
MeshSampler.data_generation(self, key)
```

Generates data containing batch_size samples.

## `TemporalMeshSampler` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/samplers.py#L91' target='_blank'>[source]</a>

*Bases: `BaseSampler`*

Infinite iterator over randomly generated batches.

The global batch is sharded across devices by the training step, so the
batch size must be divisible by the number of devices.

### `TemporalMeshSampler.data_generation()`

```python
TemporalMeshSampler.data_generation(self, key)
```

Generates data containing batch_size samples
