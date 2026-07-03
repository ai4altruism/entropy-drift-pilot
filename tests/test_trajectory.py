import math

from entropydrift.trajectory import (
    coherence,
    entropy_trajectory,
    is_monotone,
    max_positive_jump,
    shannon_entropy,
    violation_count,
)


def test_entropy_bounds():
    assert shannon_entropy([]) == 0.0
    assert shannon_entropy(["a", "a", "a"]) == 0.0
    # uniform over 4 outcomes -> ln 4 in nats
    assert math.isclose(shannon_entropy(["a", "b", "c", "d"]), math.log(4))
    # base 2 gives 2 bits for 4 uniform outcomes
    assert math.isclose(shannon_entropy(["a", "b", "c", "d"], base=2), 2.0)


def test_trajectory_from_step_answers():
    steps = [["a", "b", "c"], ["a", "a", "b"], ["a", "a", "a"]]
    traj = entropy_trajectory(steps)
    assert traj[0] > traj[1] > traj[2] == 0.0


def test_monotone_and_violations():
    assert is_monotone([1.0, 0.8, 0.8, 0.5])
    assert not is_monotone([1.0, 0.8, 0.9, 0.5])  # one spike
    assert violation_count([1.0, 0.8, 0.9, 0.5]) == 1
    assert violation_count([0.5, 0.6, 0.4, 0.7]) == 2


def test_eps_tolerance():
    # a rise within eps is tolerated
    assert is_monotone([0.5, 0.505], eps=0.01)
    assert not is_monotone([0.5, 0.52], eps=0.01)


def test_coherence_and_jump():
    assert math.isclose(coherence([1.0, 0.3]), 0.7)
    assert coherence([0.5]) == 0.0
    assert math.isclose(max_positive_jump([1.0, 0.8, 0.9, 0.5]), 0.1)
    assert max_positive_jump([1.0, 0.8, 0.5]) == 0.0
