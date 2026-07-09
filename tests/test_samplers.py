import jax.numpy as jnp
import numpy as np
import pytest

from jaxpi.samplers import MeshSampler, TemporalMeshSampler, UniformSampler


DOM = jnp.array([[0.0, 1.0], [-1.0, 2.0]])


def test_uniform_sampler_shape_bounds_and_sorting():
    sampler = UniformSampler(DOM, batch_size=64)
    batch = next(iter(sampler))

    assert batch.shape == (64, 2)
    assert (batch[:, 0] >= 0.0).all() and (batch[:, 0] <= 1.0).all()
    assert (batch[:, 1] >= -1.0).all() and (batch[:, 1] <= 2.0).all()
    # Sorted along the time axis (required for causal training)
    assert (jnp.diff(batch[:, 0]) >= 0).all()


def test_uniform_sampler_produces_fresh_batches():
    iterator = iter(UniformSampler(DOM, batch_size=64))
    first = next(iterator)
    second = next(iterator)
    assert not np.allclose(np.asarray(first), np.asarray(second))


def test_uniform_sampler_no_sorting():
    sampler = UniformSampler(DOM, batch_size=64, sort_axis=None)
    batch = next(iter(sampler))
    assert batch.shape == (64, 2)


def test_batch_size_must_divide_device_count():
    # conftest forces 8 devices
    with pytest.raises(AssertionError, match="divisible"):
        UniformSampler(DOM, batch_size=63)


def test_mesh_sampler():
    mesh = jnp.arange(20.0).reshape(10, 2)
    batch = next(iter(MeshSampler(mesh, batch_size=16)))
    assert batch.shape == (16, 2)

    # every sampled row must come from the mesh
    mesh_rows = {tuple(row) for row in np.asarray(mesh)}
    assert all(tuple(row) in mesh_rows for row in np.asarray(batch))


def test_mesh_sampler_with_labels():
    mesh = jnp.arange(20.0).reshape(10, 2)
    labels = jnp.arange(10.0)
    batch, batch_labels = next(iter(MeshSampler(mesh, labels=labels, batch_size=16)))

    assert batch.shape == (16, 2)
    assert batch_labels.shape == (16,)
    # labels must stay aligned with their mesh rows (mesh row i is [2i, 2i+1])
    np.testing.assert_allclose(np.asarray(batch[:, 0]) / 2.0, np.asarray(batch_labels))


def test_temporal_mesh_sampler():
    mesh = jnp.arange(20.0).reshape(10, 2)
    sampler = TemporalMeshSampler(jnp.array([0.0, 1.0]), mesh, batch_size=16)
    batch = next(iter(sampler))

    assert batch.shape == (16, 3)
    assert (batch[:, 0] >= 0.0).all() and (batch[:, 0] <= 1.0).all()
