import logging as py_logging
import os
import re
import json

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl import logging as absl_logging


class _DropOrbaxInfo(py_logging.Filter):
    """Drop Orbax's INFO chatter (~30 lines per async checkpoint save) while
    keeping its warnings/errors and everything else untouched."""

    def filter(self, record):
        if record.levelno >= py_logging.WARNING:
            return True
        return "orbax" not in getattr(record, "pathname", "")


# Orbax logs through absl; filter at the handler so training logs stay clean
absl_logging.get_absl_handler().addFilter(_DropOrbaxInfo())


def get_ckpt_path(config):
    """Checkpoint root for a run: <saving.ckpt_path or cwd>/<wandb.name>/ckpt."""
    base = config.saving.get("ckpt_path", None)
    base = os.getcwd() if base is None else os.path.abspath(base)
    return os.path.join(base, config.wandb.name, "ckpt")


def has_checkpoint_steps(path):
    """Whether an Orbax checkpoint directory contains at least one saved step."""
    if not os.path.isdir(path):
        return False
    return any(name.isdigit() for name in os.listdir(path))


def latest_time_window(ckpt_path, pattern=r"time_window_(\d+)"):
    """Index of the last trained time window, based on checkpoint directories."""
    if not os.path.isdir(ckpt_path):
        return 0
    indices = [0]
    for name in os.listdir(ckpt_path):
        match = re.fullmatch(pattern, name)
        if match is not None and has_checkpoint_steps(os.path.join(ckpt_path, name)):
            indices.append(int(match.group(1)))
    return max(indices)


def create_checkpoint_manager(config, ckpt_path, suffix=None):
    if suffix is not None:
        ckpt_path = os.path.join(ckpt_path, str(suffix))
    ckpt_path = os.path.abspath(ckpt_path)  # Orbax requires absolute paths

    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.num_keep_ckpts,
        create=True,
    )
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)
    return ckpt_mngr


def save_checkpoint(ckpt_mngr, state):
    # state.step is a JAX array; Orbax expects a plain int step
    step = int(state.step)
    if ckpt_mngr.latest_step() == step:  # already saved (e.g. final save)
        return
    ckpt_mngr.save(step, args=ocp.args.StandardSave(state))


def restore_checkpoint(ckpt_mngr, state, step=None):
    step = step if step is not None else ckpt_mngr.latest_step()
    if step is None:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_mngr.directory}. "
            "Train and save a checkpoint before restoring."
        )
    restored = ckpt_mngr.restore(
        step,
        args=ocp.args.StandardRestore(state),
    )
    # Orbax restores arrays onto a single device; replicate them across all
    # devices to match the sharded training step.
    mesh = jax.make_mesh(
        (jax.device_count(),), ("batch",),
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    restored = jax.tree.map(lambda x: jax.device_put(x, sharding), restored)
    return restored


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Custom serialization for JAX numpy arrays
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()  # Convert JAX numpy array to a list
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_config(config, workdir, name=None):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Set default name if not provided
    if name is None:
        name = "config"
    # Correctly append the '.json' extension to the filename
    config_path = os.path.join(workdir, name + '.json')

    # Write the config to a JSON file
    with open(config_path, 'w') as config_file:
        json.dump(config.to_dict(), config_file, cls=CustomJSONEncoder, indent=4)
