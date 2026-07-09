import json
import os

import jax
import numpy as np
import pytest

from jaxpi.checkpointing import (
    create_checkpoint_manager,
    restore_checkpoint,
    save_checkpoint,
    save_config,
)

from helpers import make_config, make_model


def test_save_restore_roundtrip(tmp_path):
    config = make_config()
    model = make_model(config)
    state = model.state

    manager = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    save_checkpoint(manager, state)
    manager.wait_until_finished()

    restored = restore_checkpoint(manager, state)
    for saved, loaded in zip(
        jax.tree.leaves(state.params), jax.tree.leaves(restored.params)
    ):
        np.testing.assert_allclose(np.asarray(saved), np.asarray(loaded))


def test_relative_ckpt_path_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config()
    manager = create_checkpoint_manager(config.saving, "ckpt_rel", suffix="window_0")
    assert os.path.isabs(str(manager.directory))
    assert str(manager.directory).startswith(str(tmp_path))


def test_restore_from_empty_dir_raises(tmp_path):
    config = make_config()
    model = make_model(config)
    manager = create_checkpoint_manager(config.saving, str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        restore_checkpoint(manager, model.state)


def test_save_config(tmp_path):
    config = make_config()
    save_config(config, str(tmp_path / "workdir"))
    with open(tmp_path / "workdir" / "config.json") as f:
        loaded = json.load(f)
    assert loaded["seed"] == 0
    assert loaded["arch"]["arch_name"] == "Mlp"
