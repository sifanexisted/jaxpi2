import itertools
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, jit


class BaseSampler:
    """Infinite iterator over randomly generated batches.

    The global batch is sharded across devices by the training step, so the
    batch size must be divisible by the number of devices.
    """

    _uid_counter = itertools.count()

    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        # data_generation is jitted with static self; see PINN.__init__ for
        # why the default id()-based hash is unsafe across instances.
        self._uid = next(BaseSampler._uid_counter)

        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = jax.device_count()
        assert batch_size % self.num_devices == 0, (
            f"Batch size {batch_size} must be divisible by the number of "
            f"devices {self.num_devices}"
        )

    def __hash__(self):
        return self._uid

    def __eq__(self, other):
        return self is other

    def __iter__(self):
        return self

    def __next__(self):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        batch = self.data_generation(subkey)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, sort_axis=0, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]
        self.sort_axis = sort_axis  # sorting by the first coordinate (e.g., time) for causal training

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        if self.sort_axis is not None:
            sorted_indices = jnp.argsort(batch[:, self.sort_axis])
            batch = batch[sorted_indices]
        return batch


class MeshSampler(BaseSampler):
    def __init__(self, mesh, labels=None, batch_size=1024, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.mesh = mesh
        self.labels = labels

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        """Generates data containing batch_size samples."""
        idx = random.choice(key, self.mesh.shape[0], shape=(self.batch_size,))
        batch = self.mesh[idx, :]

        if self.labels is None:
            return batch
        else:
            batch_labels = self.labels[idx]
            return batch, batch_labels


class TemporalMeshSampler(BaseSampler):
    def __init__(
            self, temporal_dom, mesh, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)
        self.temporal_dom = temporal_dom
        self.mesh = mesh

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        temporal_batch = random.uniform(
            key1,
            shape=(self.batch_size, 1),
            minval=self.temporal_dom[0],
            maxval=self.temporal_dom[1],
        )
        spatial_idx = random.choice(
            key2, self.mesh.shape[0], shape=(self.batch_size,)
        )
        spatial_batch = self.mesh[spatial_idx, :]
        batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)

        return batch
