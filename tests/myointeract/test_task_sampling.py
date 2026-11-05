import jax
import jax.numpy as jp
import numpy as np
import pytest

from myosuite.envs.myo.myouser.myouser_universal import (
    DefaultTargetsConfig, PointingTarget, PointingTargetClass, TenPointingTargetsConfig,
)
from myosuite.envs.myo.myouser.target_validation import (
    normalize_pointing_ranges, validate_target_sampling,
)


def test_rejects_impossible_and_seed_dependent_target_pairs():
    fixed = PointingTarget(position=[0.3, 0, 0], size=0.1)
    with pytest.raises(ValueError, match="cannot always be placed"):
        validate_target_sampling([fixed, fixed], 0.05)
    broad = PointingTarget(position=[[-1, -1, -1], [1, 1, 1]], size=0.01)
    narrow = PointingTarget(position=[[-0.01, 0, 0], [0.01, 0, 0]], size=0.1)
    with pytest.raises(ValueError, match="cannot always be placed"):
        validate_target_sampling([broad, narrow], 0.05)


@pytest.mark.parametrize("targets", [DefaultTargetsConfig(), TenPointingTargetsConfig()])
def test_shipped_target_templates_are_feasible(targets):
    validate_target_sampling(targets.targets, 0.05)


def test_validates_ranges_dwell_and_documented_fixed_values():
    bounds, radii = normalize_pointing_ranges([0.3, 0, 0], 0.1)
    np.testing.assert_array_equal(bounds, [[0.3, 0, 0], [0.3, 0, 0]])
    np.testing.assert_array_equal(radii, [0.1, 0.1])
    with pytest.raises(ValueError, match="lower bounds"):
        normalize_pointing_ranges([[1, 0, 0], [0, 0, 0]], 0.1)
    with pytest.raises(ValueError, match="positive and ordered"):
        normalize_pointing_ranges([0, 0, 0], [-1, 0.1])
    with pytest.raises(ValueError, match="dwell duration"):
        validate_target_sampling([PointingTarget(dwell_duration=0)], 0.05)


def test_bounded_sampler_fallback_preserves_separation_under_jit_vmap(monkeypatch):
    target = PointingTargetClass(
        1, 2, [[-1, -1, -1], [1, 1, 1]], [0.1, 0.5], 5, jp.zeros(3), {}
    )
    # Force rejection of every random candidate to exercise the finite fallback.
    def reject(info):
        return {**info, "sampled_pos": info["prev_pos"], "attempts": info["attempts"] + 1}
    monkeypatch.setattr(target, "sampler", reject)
    previous = jp.array([[0., 0., 0.], [0.3, -0.4, 0.2]])
    keys = jax.random.split(jax.random.PRNGKey(7), len(previous))
    sample = jax.jit(jax.vmap(target.generate_target_pos))
    positions = sample(keys, previous)
    assert np.all(np.linalg.norm(positions - previous, axis=-1) >= 1.0)
    assert np.all(np.abs(positions) <= 1.0)
    np.testing.assert_array_equal(positions, sample(keys, previous))


def test_regular_sampling_is_seeded_and_separated_under_jit_vmap():
    target = PointingTargetClass(
        1, 2, [[-1, -1, -1], [1, 1, 1]], [0.1, 0.5], 5, jp.zeros(3), {}
    )
    keys = jax.random.split(jax.random.PRNGKey(13), 16)
    previous = jp.zeros((16, 3))
    sample = jax.jit(jax.vmap(target.generate_target_pos))
    positions = sample(keys, previous)
    assert np.all(np.linalg.norm(positions - previous, axis=-1) >= 1.0)
    np.testing.assert_array_equal(positions, sample(keys, previous))
