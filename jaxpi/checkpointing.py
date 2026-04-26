import os
import json

import jax.numpy as jnp
import orbax.checkpoint as ocp


def create_checkpoint_manager(config, ckpt_path, suffix=None):
    if suffix is not None:
        ckpt_path = os.path.join(ckpt_path, str(suffix))

    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.num_keep_ckpts,
        create=True,
    )
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)
    return ckpt_mngr


def save_checkpoint(ckpt_mngr, state):
    ckpt_mngr.save(state.step, args=ocp.args.StandardSave(state))


def restore_checkpoint(ckpt_mngr, state, step=None):
    step = step if step is not None else ckpt_mngr.latest_step()
    restored = ckpt_mngr.restore(
        step,
        args=ocp.args.StandardRestore(state),
    )
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
