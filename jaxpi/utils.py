from typing import Callable

from jax.flatten_util import ravel_pytree


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


def create_update_scheduler(
        every: int = 100,
        start: int = 0,
) -> Callable[[int], bool]:
    """Build and return a step-checker for the given schedule."""
    def should_update(step: int) -> bool:
        s = step - start
        return s >= 0 and (s % every) == 0

    return should_update
