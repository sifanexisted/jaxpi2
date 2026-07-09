from typing import Callable

from jax.flatten_util import ravel_pytree

import optax


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


def _find_schedule_free_state(opt_state):
    """Locate the ScheduleFreeState inside a (possibly chained) optax state."""
    if hasattr(opt_state, "b1") and hasattr(opt_state, "z"):
        return opt_state
    if isinstance(opt_state, (tuple, list)):
        for inner in opt_state:
            found = _find_schedule_free_state(inner)
            if found is not None:
                return found
    return None


def get_eval_params(state, schedule_free):
    """Parameters to use for inference.

    With the schedule-free optimizer wrapper, the parameters held by the train
    state are the training iterates; evaluation (error logging, IC propagation
    between time windows, final prediction) should use the schedule-free
    averaged parameters instead.
    """
    if schedule_free:
        sf_state = _find_schedule_free_state(state.opt_state)
        if sf_state is None:
            raise ValueError("No schedule-free state found in the optimizer state")
        return optax.contrib.schedule_free_eval_params(sf_state, state.params)
    return state.params


def create_update_scheduler(
        every: int = 100,
        start: int = 0,
) -> Callable[[int], bool]:
    """Build and return a step-checker for the given schedule."""
    def should_update(step: int) -> bool:
        s = step - start
        return s >= 0 and (s % every) == 0

    return should_update
