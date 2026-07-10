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
from jaxpi.training import _ensure_wandb, sample_batches, train_loop, train_time_windows

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
        losses.append(log_dict["loss/ics"] + log_dict["loss/res"])
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


def test_train_loop_resume_keeps_pts_reference(tmp_path, batches):
    """The pseudo-time reference (init_state) must be the run's initial state
    even when resuming from a checkpoint, not the restored parameters —
    otherwise the shrink schedule resets and the weights blow up."""
    config = small_config(pseudo_time=True)
    config.saving.save_every_steps = 10
    model = make_model(config)

    ckpt_mngr = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    train_loop(config, model, batches, ckpt_mngr=ckpt_mngr)
    ckpt_mngr.wait_until_finished()

    # Resume with more steps; capture the init_state passed to the weight update
    config.training.resume = True
    config.training.max_steps = 40
    model2 = make_model(config)
    fresh_params = jax.tree.leaves(model2.state.params)

    captured = []
    original = model2.update_pts_weights
    model2.update_pts_weights = lambda state, init_state, batch: (
        captured.append(init_state) or original(state, init_state, batch)
    )

    ckpt_mngr2 = create_checkpoint_manager(config.saving, str(tmp_path / "ckpt"))
    train_loop(config, model2, batches, ckpt_mngr=ckpt_mngr2)

    assert captured, "no pts weight update fired after resume"
    assert int(captured[0].step) == 0, "init_state must be the initial state"
    for ref, fresh in zip(jax.tree.leaves(captured[0].params), fresh_params):
        np.testing.assert_array_equal(np.asarray(ref), np.asarray(fresh))


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

    # Train two windows, then "restart" asking for 3 total: the run must
    # resume after window 2 and train only window 3.
    model = make_model(config)
    train_time_windows(config, model, make_samplers)

    config.training.resume = True
    config.training.num_time_windows = 3
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


def test_train_time_windows_resume_mid_window(tmp_path, monkeypatch):
    """A window interrupted mid-training is re-entered and finished, not
    skipped in favor of the next window."""
    monkeypatch.chdir(tmp_path)
    config = small_config()

    def make_samplers(window_idx):
        def gen():
            key = jax.random.PRNGKey(window_idx)
            while True:
                key, subkey = jax.random.split(key)
                yield make_batch(subkey)

        return gen()

    # "Interrupt" window 2 by training it with fewer steps than the resumed
    # run will ask for (window 1 completes at 30 steps either way).
    model = make_model(config)
    train_time_windows(config, model, make_samplers)  # windows 1, 2 @ 30 steps

    config.training.resume = True
    config.training.max_steps = 50  # window 2's 30-step checkpoint is partial
    propagated = []
    model2 = make_model(config)
    train_time_windows(
        config, model2, make_samplers,
        propagate_ic=lambda m, idx: propagated.append(idx),
    )

    ckpt_path = get_ckpt_path(config)
    # Window 2 was finished to 50 steps; no window 3 was started
    mngr = create_checkpoint_manager(
        config.saving, ckpt_path, suffix="time_window_2"
    )
    assert mngr.latest_step() == 50
    assert latest_time_window(ckpt_path) == 2
    assert int(model2.state.step) == 50
    # propagate_ic: once for the IC of re-entered window 2 (idx 0), once
    # after window 2 finishes (idx 1)
    assert propagated == [0, 1]


def _window_samplers(window_idx):
    def gen():
        key = jax.random.PRNGKey(window_idx)
        while True:
            key, subkey = jax.random.split(key)
            yield make_batch(subkey)

    return gen()


def test_train_time_windows_step_offsets_on_resume(tmp_path, monkeypatch, offline_wandb):
    """A resumed run must log at exactly the W&B steps where the previous
    session left off: no re-logging of finished windows, no gaps or jumps."""
    monkeypatch.chdir(tmp_path)
    config = small_config()  # max_steps=30, log_every=10

    model = make_model(config)
    train_time_windows(config, model, _window_samplers)
    # window 1 at offset 0, window 2 at offset 30
    assert offline_wandb == [0, 10, 20, 30, 40, 50]

    session_a = len(offline_wandb)
    config.training.resume = True
    config.training.num_time_windows = 3
    model2 = make_model(config)
    train_time_windows(config, model2, _window_samplers)

    # Only window 3 is trained, logged at offset 2 * 30
    assert offline_wandb[session_a:] == [60, 70, 80]


def test_train_time_windows_mid_window_resume_step_offset(
    tmp_path, monkeypatch, offline_wandb
):
    """Re-entering an interrupted window logs from (window offset + restored
    step): the finished part of the window and earlier windows are not
    re-trained or re-logged."""
    monkeypatch.chdir(tmp_path)
    config = small_config()

    model = make_model(config)
    train_time_windows(config, model, _window_samplers)  # windows 1, 2 @ 30 steps
    session_a = len(offline_wandb)

    # Window 2's 30-step checkpoint is partial for a 50-step run
    config.training.resume = True
    config.training.max_steps = 50
    model2 = make_model(config)
    train_time_windows(config, model2, _window_samplers)

    # Window 2 resumes at step 30 with offset 1 * 50: steps 80, 90 only
    assert offline_wandb[session_a:] == [80, 90]


def test_train_time_windows_logs_window_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = small_config()

    logged = []
    monkeypatch.setattr(
        wandb, "log", lambda log_dict, step=None: logged.append((step, log_dict))
    )

    model = make_model(config)
    train_time_windows(config, model, _window_samplers)

    windows = [d["time_window"] for _, d in logged]
    assert windows == [1, 1, 1, 2, 2, 2]


class _FakeRun:
    def __init__(self, run_id):
        self.id = run_id


def test_ensure_wandb_resumes_same_run(tmp_path, monkeypatch):
    """With resume enabled, a restarted session reuses the persisted W&B run
    id (continuing the same run) instead of creating a fresh run that starts
    mid-axis; without resume, a new id is generated and persisted."""
    init_calls = []
    counter = itertools.count(1)

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        run_id = kwargs.get("id") or f"generated-{next(counter)}"
        wandb.run = _FakeRun(run_id)

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(jax, "process_index", lambda: 0)

    config = small_config()
    run_dir = str(tmp_path / "test-run")
    id_file = os.path.join(run_dir, "wandb_id.txt")

    # First session: new run, id persisted
    _ensure_wandb(config, run_dir=run_dir)
    assert init_calls[-1]["id"] is None and init_calls[-1]["resume"] is None
    assert open(id_file).read() == "generated-1"

    # Restart with resume: the persisted id is reused
    wandb.run = None
    config.training.resume = True
    _ensure_wandb(config, run_dir=run_dir)
    assert init_calls[-1]["id"] == "generated-1"
    assert init_calls[-1]["resume"] == "allow"
    assert open(id_file).read() == "generated-1"

    # Fresh run without resume: stale id ignored and overwritten
    wandb.run = None
    config.training.resume = False
    _ensure_wandb(config, run_dir=run_dir)
    assert init_calls[-1]["id"] is None
    assert open(id_file).read() == "generated-2"

    # Already-initialized run: no double init
    n = len(init_calls)
    _ensure_wandb(config, run_dir=run_dir)
    assert len(init_calls) == n


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
