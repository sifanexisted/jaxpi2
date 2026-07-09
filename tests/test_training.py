import itertools
import os

import jax
import numpy as np
import pytest
import wandb

from jaxpi.checkpointing import (
    create_checkpoint_manager,
    get_ckpt_path,
    has_checkpoint_steps,
    latest_time_window,
)
from jaxpi.training import sample_batches, train_loop, train_time_windows

from helpers import make_batch, make_config, make_model


@pytest.fixture(autouse=True)
def offline_wandb(monkeypatch, tmp_path):
    """Stub out W&B so training tests never talk to the network."""
    logged = []
    monkeypatch.setattr(wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb, "log", lambda log_dict, step=None: logged.append(step))
    monkeypatch.setattr(wandb, "run", object())  # pretend a run exists
    return logged


@pytest.fixture
def batches():
    def gen():
        key = jax.random.PRNGKey(0)
        while True:
            key, subkey = jax.random.split(key)
            yield make_batch(subkey)

    return gen()


def small_config(**kwargs):
    return make_config(**kwargs)


def test_train_loop_reduces_loss_and_checkpoints(tmp_path, batches, offline_wandb):
    config = small_config()
    config.training.max_steps = 60
    model = make_model(config)

    losses = []

    def spy(step, log_dict):
        losses.append(log_dict["ics_loss"] + log_dict["res_loss"])
        return False

    ckpt_mngr = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    train_loop(config, model, batches, ckpt_mngr=ckpt_mngr, stop_fn=spy)

    assert int(model.state.step) == config.training.max_steps
    assert losses[-1] < losses[0]
    assert ckpt_mngr.latest_step() == config.training.max_steps
    # wandb logged every log_every_steps
    assert offline_wandb == list(range(0, config.training.max_steps, 10))


def test_train_loop_weight_updates_fire_on_schedule(batches):
    config = small_config(pseudo_time=True)
    model = make_model(config)
    initial_loss_weights = dict(model.state.loss_weights)
    initial_pts_weights = dict(model.state.pts_weights)

    train_loop(config, model, batches)

    # Update schedule (start=5, every=10) fires at steps 5, 15, 25
    for key, w0 in initial_loss_weights.items():
        assert not np.allclose(np.asarray(model.state.loss_weights[key]), w0)
    for key, w0 in initial_pts_weights.items():
        assert not np.allclose(np.asarray(model.state.pts_weights[key]), w0)


def test_train_loop_early_stop(tmp_path, batches):
    config = small_config()
    model = make_model(config)

    ckpt_mngr = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    train_loop(
        config, model, batches, ckpt_mngr=ckpt_mngr,
        stop_fn=lambda step, log_dict: step >= 10,
    )

    assert int(model.state.step) == 11  # stopped at step 10 (0-based)
    assert ckpt_mngr.latest_step() == 11  # checkpoint written on early stop


def test_train_loop_accepts_samplers_and_fixed_batches(batches):
    config = small_config()
    config.training.max_steps = 3

    # dict of "samplers" (any iterables work)
    model = make_model(config)
    fixed = make_batch(jax.random.PRNGKey(1))
    train_loop(config, model, itertools.repeat(fixed))
    assert int(model.state.step) == 3

    # sample_batches over a dict yields dict batches
    it = sample_batches({"a": itertools.repeat(1), "b": itertools.repeat(2)})
    assert next(it) == {"a": 1, "b": 2}


def test_train_loop_resume(tmp_path, batches):
    config = small_config()
    model = make_model(config)

    ckpt_mngr = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    train_loop(config, model, batches, ckpt_mngr=ckpt_mngr)
    ckpt_mngr.wait_until_finished()

    # Fresh model resumes from the saved step and has nothing left to do
    config.training.resume = True
    model2 = make_model(config)
    ckpt_mngr2 = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    train_loop(config, model2, batches, ckpt_mngr=ckpt_mngr2)

    assert int(model2.state.step) == config.training.max_steps
    for restored, trained in zip(
        jax.tree.leaves(model2.state.params), jax.tree.leaves(model.state.params)
    ):
        np.testing.assert_allclose(np.asarray(restored), np.asarray(trained))


def test_train_time_windows_hooks_and_transfer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = small_config()
    model = make_model(config)

    calls = {"samplers": [], "propagate": [], "eval_args": []}

    def make_samplers(window_idx):
        calls["samplers"].append(window_idx)

        def gen():
            key = jax.random.PRNGKey(window_idx)
            while True:
                key, subkey = jax.random.split(key)
                yield make_batch(subkey)

        return gen()

    def propagate_ic(model, window_idx):
        calls["propagate"].append(window_idx)

    def make_eval_args(window_idx):
        calls["eval_args"].append(window_idx)
        return ()

    train_time_windows(
        config, model, make_samplers,
        propagate_ic=propagate_ic, make_eval_args=make_eval_args,
    )

    assert calls["samplers"] == [0, 1]
    assert calls["propagate"] == [0, 1]
    assert calls["eval_args"] == [0, 1]

    # Each window checkpoints under its own suffix
    ckpt_path = get_ckpt_path(config)
    assert latest_time_window(ckpt_path) == 2

    # Second window starts from a fresh optimizer: step counts restart
    assert int(model.state.step) == config.training.max_steps


def test_train_time_windows_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = small_config()

    def make_samplers(window_idx):
        def gen():
            key = jax.random.PRNGKey(window_idx)
            while True:
                key, subkey = jax.random.split(key)
                yield make_batch(subkey)

        return gen()

    # Train two windows, then "restart" with 1 more window: the run must
    # resume from window 2 and train only window 3.
    model = make_model(config)
    train_time_windows(config, model, make_samplers)

    config.training.resume = True
    config.training.num_time_windows = 1
    propagated = []
    model2 = make_model(config)
    train_time_windows(
        config, model2, make_samplers,
        propagate_ic=lambda m, idx: propagated.append(idx),
    )

    ckpt_path = get_ckpt_path(config)
    assert latest_time_window(ckpt_path) == 3
    # propagate_ic called once after the resume restore (window 1) and once
    # after the newly trained window (window 2, 0-based)
    assert propagated == [1, 2]


def test_checkpoint_helpers(tmp_path):
    ckpt_path = tmp_path / "ckpt"
    assert latest_time_window(str(ckpt_path)) == 0

    # Synthetic layout: window 1 has a step, window 2 is empty, junk ignored
    (ckpt_path / "time_window_1" / "10").mkdir(parents=True)
    (ckpt_path / "time_window_2").mkdir()
    (ckpt_path / "notes.txt").parent.mkdir(exist_ok=True)
    (ckpt_path / "notes.txt").write_text("junk")

    assert has_checkpoint_steps(str(ckpt_path / "time_window_1"))
    assert not has_checkpoint_steps(str(ckpt_path / "time_window_2"))
    assert latest_time_window(str(ckpt_path)) == 1


def test_get_ckpt_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = small_config()
    assert get_ckpt_path(config) == os.path.join(str(tmp_path), "test-run", "ckpt")
