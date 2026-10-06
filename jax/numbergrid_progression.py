"""One-time map preparation checks; never called by a JAX rollout or PPO update."""
import random


def validate_progression(numbers, initial_number, minimum_number):
    """Check numeric reachability under strict '<' captures and +1 growth.

    Taking enemies in ascending order reaches the maximum possible number.
    This checks values only; the map's legal spatial route is tested separately.
    """
    number = initial_number
    for enemy in sorted(numbers):
        if enemy >= number:
            break
        number += 1
    if number < minimum_number:
        raise ValueError(
            f'Enemy numbers block progression: start {initial_number}, '
            f'maximum reachable {number}, required {minimum_number}'
        )
    return number


def generate_opponent_numbers(seed):
    """Prepare twelve fixed values in 1..12, including 1/12, with no numeric gap.

    Reject impossible draws instead of changing contact or reward rules. This
    runs once when authoring a map, never at environment reset or training time.
    """
    rng = random.Random(seed)
    for _ in range(10_000):
        numbers = rng.choices(range(1, 13), k=12)
        numbers[0], numbers[-1] = 1, 12
        try:
            validate_progression(numbers, initial_number=2, minimum_number=14)
        except ValueError:
            continue
        return numbers
    raise RuntimeError('Could not generate a valid progression; no map was accepted')
