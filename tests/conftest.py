import os

# Run the tests on CPU with 8 fake devices so that the sharded (multi-GPU)
# code paths in jaxpi.models are exercised deterministically. Must be set
# before jax is imported anywhere; deliberately overrides any inherited
# JAX_PLATFORMS (on GPU, TF32 matmuls break the exact-equivalence tests).
os.environ["JAX_PLATFORMS"] = "cpu"
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (
        _flags + " --xla_force_host_platform_device_count=8"
    ).strip()
