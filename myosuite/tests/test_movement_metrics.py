"""Validation of the movement-quality metrics against analytic and published values.

Analytic minimum-jerk properties (Flash & Hogan 1985), the reference SPARC/LDLJ code
of Balasubramanian et al. (2015), Fitts' law quantities with known answers (also over
ISO 9241-411 style conditions of pointing trials), and the two-thirds power law on a
harmonic ellipse. No RL; synthetic trajectories only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, quad

from myosuite.utils import movement_metrics as mm
from myosuite.utils.min_jerk import _min_jerk_spaces, generate_joint_space_min_jerk

# Minimum-jerk onset at 5% of peak speed: 30 tau^2 (1 - tau)^2 = 0.05 * 1.875.
_TAU_ONSET = (1 - math.sqrt(1 - 4 * math.sqrt(0.05 * 1.875 / 30))) / 2


def _min_jerk_samples(n: int = 1001, duration: float = 1.0, distance: float = 0.3):
    t = np.linspace(0.0, duration, n)
    return t, t[1] - t[0], mm.minimum_jerk(0.0, distance, duration, t)


# ── Minimum jerk ─────────────────────────────────────────────────────────────


def test_minimum_jerk_boundary_conditions_and_peak():
    start, end = np.array([0.1, -0.2, 0.3]), np.array([0.4, 0.2, 0.3])
    t = np.array([-0.1, 0.0, 0.25, 0.5, 1.0, 1.2])
    mj = mm.minimum_jerk(start, end, 0.5, t)
    np.testing.assert_allclose(mj.position[[0, 1]], [start, start])
    np.testing.assert_allclose(mj.position[[4, 5]], [end, end])
    np.testing.assert_allclose(mj.velocity[[1, 4]], 0.0, atol=1e-12)
    np.testing.assert_allclose(mj.acceleration[[1, 4]], 0.0, atol=1e-12)
    np.testing.assert_allclose(mj.position[2], (start + end) / 2)  # symmetric
    peak = np.linalg.norm(mj.velocity[2])
    assert peak == pytest.approx(1.875 * np.linalg.norm(end - start) / 0.5)


def test_minimum_jerk_sampled_peak_speed_and_symmetry():
    _, dt, mj = _min_jerk_samples()
    v = mm.speed(mj.position, dt)
    assert v.max() == pytest.approx(mm.MIN_JERK_PEAK_SPEED_FACTOR * 0.3 / 1.0, rel=1e-9)
    bounds = mm.movement_bounds(v, dt)
    assert mm.time_to_peak_ratio(v, dt, bounds) == pytest.approx(0.5, abs=1e-9)


def test_minimum_jerk_squared_jerk_constant_is_720():
    integral, _ = quad(lambda tau: (60 - 360 * tau + 360 * tau**2) ** 2, 0, 1)
    assert integral == pytest.approx(720.0, rel=1e-12)


@pytest.mark.parametrize("n", [101, 1001])
def test_minimum_jerk_dimensionless_jerk_and_ldlj(n):
    _, dt, mj = _min_jerk_samples(n)
    rel = 1e-6 if n == 1001 else 1e-5
    dj_amp = mm.dimensionless_jerk(mj.position, dt, normalization="amplitude")
    dj_peak = mm.dimensionless_jerk(mj.position, dt, normalization="peak_speed")
    assert dj_amp == pytest.approx(720.0, rel=rel)
    assert dj_peak == pytest.approx(720.0 / 1.875**2, rel=rel)  # 204.8
    assert mm.log_dimensionless_jerk(mj.position, dt) == pytest.approx(
        -math.log(204.8), abs=rel
    )
    assert mm.log_dimensionless_jerk(
        mj.position, dt, normalization="amplitude"
    ) == pytest.approx(-math.log(720.0), abs=rel)


def test_dimensionless_jerk_is_scale_and_direction_invariant():
    values = []
    for distance, duration, direction in [
        (0.05, 0.4, [1.0, 0.0, 0.0]),
        (2.0, 3.0, [0.3, -0.5, 0.8]),
    ]:
        t = np.linspace(0.0, duration, 401)
        end = distance * np.asarray(direction) / np.linalg.norm(direction)
        mj = mm.minimum_jerk(np.zeros(3), end, duration, t)
        values.append(mm.dimensionless_jerk(mj.position, t[1]))
    assert values[0] == pytest.approx(values[1], rel=1e-6)
    assert values[0] == pytest.approx(204.8, rel=1e-5)


def test_savgol_derivatives_are_exact_for_minimum_jerk():
    _, dt, mj = _min_jerk_samples(201)
    dj = mm.dimensionless_jerk(
        mj.position, dt, normalization="amplitude", savgol=(11, 5)
    )
    assert dj == pytest.approx(720.0, rel=1e-5)


# ── Reference implementation (Balasubramanian et al. 2015) ───────────────────


def test_sparc_and_ldlj_match_reference_code():
    # Doctests of github.com/siva82kb/SPARC scripts/smoothness.py:
    # Gaussian speed profile, fs = 100 Hz -> SPARC -1.41403, LDLJ -5.81636.
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5 * t**2)
    assert mm.sparc(move, 0.01) == pytest.approx(-1.41403, abs=1e-5)
    position = cumulative_trapezoid(move, dx=0.01, initial=0.0)
    ldlj = mm.log_dimensionless_jerk(position, 0.01)
    # The reference uses T = N dt; this module T = (N - 1) dt.
    n = len(move)
    assert ldlj - 3 * math.log(n / (n - 1)) == pytest.approx(-5.81636, abs=1e-3)


def test_smoothness_drops_with_a_second_submovement():
    t = np.arange(0.0, 2.0, 0.01)
    one = mm.minimum_jerk(0.0, 0.1, 1.0, t).position
    two = one + mm.minimum_jerk(0.0, 0.1, 1.0, t - 0.7).position
    v_one, v_two = mm.speed(one, 0.01), mm.speed(two, 0.01)
    assert mm.count_speed_peaks(v_one) == 1
    assert mm.count_speed_peaks(v_two) == 2
    assert mm.log_dimensionless_jerk(two, 0.01) < mm.log_dimensionless_jerk(one, 0.01)
    assert mm.sparc(v_two, 0.01) < mm.sparc(v_one, 0.01)


def test_sparc_is_amplitude_invariant():
    _, dt, mj = _min_jerk_samples(501)
    v = mm.speed(mj.position, dt)
    assert mm.sparc(3.7 * v, dt) == pytest.approx(mm.sparc(v, dt), abs=1e-12)


# ── Kinematics and segmentation ──────────────────────────────────────────────


def test_onset_offset_and_movement_time_noise_free():
    _, dt, mj = _min_jerk_samples()
    bounds = mm.movement_bounds(mm.speed(mj.position, dt), dt)
    assert bounds.onset_time == pytest.approx(_TAU_ONSET, abs=1e-5)
    assert bounds.offset_time == pytest.approx(1 - _TAU_ONSET, abs=1e-5)
    assert bounds.movement_time == pytest.approx(1 - 2 * _TAU_ONSET, abs=2e-5)


def test_speed_and_onset_detection_on_noisy_minimum_jerk():
    rng = np.random.default_rng(0)
    dt, start_time, duration = 0.01, 0.25, 1.0
    t = np.arange(0.0, 1.5 + 1e-9, dt)
    end = np.array([0.2, 0.1, 0.0])
    mj = mm.minimum_jerk(np.zeros(3), end, duration, t - start_time)
    noisy = mj.position + rng.normal(0.0, 1e-4, mj.position.shape)
    v = mm.speed(noisy, dt, savgol=(21, 3))
    bounds = mm.movement_bounds(v, dt)
    assert bounds.onset_time == pytest.approx(start_time + _TAU_ONSET, abs=5e-3)
    assert bounds.offset_time == pytest.approx(
        start_time + duration * (1 - _TAU_ONSET), abs=5e-3
    )
    assert v.max() == pytest.approx(1.875 * np.linalg.norm(end) / duration, rel=0.01)


def test_movement_bounds_without_movement():
    assert mm.movement_bounds(np.zeros(10), 0.01) is None


def test_straightness_and_path_length():
    s = np.linspace(0.0, 1.0, 50) ** 2  # non-uniform speed along a line
    line = np.outer(s, [0.3, -0.4, 0.0])
    assert mm.straightness(line) == pytest.approx(1.0, abs=1e-12)
    assert mm.path_length(line) == pytest.approx(0.5)
    theta = np.linspace(0.0, np.pi, 2001)
    semicircle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    assert mm.straightness(semicircle) == pytest.approx(np.pi / 2, rel=1e-6)


def test_target_acquisition_entries_and_dwell():
    inside = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    acq = mm.target_acquisition(inside, 0.1)
    assert acq.entry_time == pytest.approx(0.2)
    assert acq.acquisition_time == pytest.approx(1.0)  # final stay, held to the end
    assert acq.entries == 3  # two re-entries (TRE = 2)
    dwell = mm.target_acquisition(inside, 0.1, dwell_time=0.3)
    assert dwell.acquisition_time == pytest.approx(0.5)  # first stay of >= 3 samples
    missed = mm.target_acquisition(inside[:-2], 0.1)
    assert math.isnan(missed.acquisition_time) and missed.entries == 2
    never = mm.target_acquisition(np.zeros(5, dtype=bool), 0.1)
    assert math.isnan(never.entry_time) and never.entries == 0


# ── Fitts' law ───────────────────────────────────────────────────────────────


def test_index_of_difficulty_shannon_form():
    np.testing.assert_allclose(mm.index_of_difficulty([3.0, 7.0], 1.0), [2.0, 3.0])


def test_effective_width_of_gaussian_endpoints():
    rng = np.random.default_rng(1)
    n, sd, distance, overshoot = 20000, 0.01, 0.2, 0.002
    angle = rng.uniform(0.0, 2 * np.pi, n)  # multi-directional task axes
    axis = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    normal = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
    starts = rng.normal(0.0, 0.05, (n, 2))
    targets = starts + distance * axis
    along = rng.normal(overshoot, sd, n)
    across = rng.normal(0.0, 5 * sd, n)  # off-axis scatter must not count
    endpoints = targets + along[:, None] * axis + across[:, None] * normal
    eff = mm.effective_parameters(starts, targets, endpoints)
    assert eff.width == pytest.approx(4.133 * sd, rel=0.02)
    assert eff.distance == pytest.approx(distance + overshoot, abs=3 * sd / np.sqrt(n))
    assert eff.index_of_difficulty == pytest.approx(
        math.log2(eff.distance / eff.width + 1)
    )


def test_effective_parameters_hand_computed():
    # 1-D task: start 0, target 10, endpoints 9, 10, 11 -> dx = -1, 0, 1 (SD 1).
    eff = mm.effective_parameters(0.0, 10.0, np.array([9.0, 10.0, 11.0]))
    assert eff.distance == pytest.approx(10.0)
    assert eff.width == pytest.approx(4.133)
    assert eff.index_of_difficulty == pytest.approx(math.log2(10.0 / 4.133 + 1.0))


def test_throughput_hand_computed():
    # IDe / MT = 4, 5, 5 bits/s -> TP = 14 / 3.
    assert mm.throughput([2.0, 3.0, 4.0], [0.5, 0.6, 0.8]) == pytest.approx(14 / 3)


def test_fitts_regression_recovers_known_line():
    ids = mm.index_of_difficulty(
        np.repeat([0.1, 0.2, 0.4], 3), np.tile([0.01, 0.02, 0.04], 3)
    )
    a, b = 0.12, 0.165
    exact = mm.fitts_regression(ids, a + b * ids)
    assert exact.intercept == pytest.approx(a)
    assert exact.slope == pytest.approx(b)
    assert exact.r_squared == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    ids_rep = np.repeat(ids, 200)
    noisy = mm.fitts_regression(
        ids_rep, a + b * ids_rep + rng.normal(0, 0.02, ids_rep.size)
    )
    assert noisy.intercept == pytest.approx(a, abs=0.01)
    assert noisy.slope == pytest.approx(b, abs=0.003)
    assert 0.9 < noisy.r_squared < 1.0


# ── Two-thirds power law ─────────────────────────────────────────────────────


def test_two_thirds_power_law_on_harmonic_ellipse():
    # x = a cos(wt), y = b sin(wt) obeys v = w (ab)^(1/3) kappa^(-1/3) exactly.
    w, a, b = 2 * np.pi, 0.2, 0.1
    t = np.linspace(0.0, 1.0, 2001)
    xy = np.stack([a * np.cos(w * t), b * np.sin(w * t)], axis=1)
    fit = mm.two_thirds_power_law(xy, t[1])
    assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert fit.gain == pytest.approx(w * (a * b) ** (1 / 3), rel=1e-5)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


# ── Trial summary ────────────────────────────────────────────────────────────


def test_point_to_point_metrics_on_two_point_minimum_jerk():
    dt, duration = 0.01, 0.8
    t = np.arange(0.0, 1.2 + 1e-9, dt)
    starts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    targets = starts + np.array([[0.2, 0.0, 0.0], [0.0, 0.15, 0.0]])
    positions = np.stack(
        [mm.minimum_jerk(s, g, duration, t).position for s, g in zip(starts, targets)],
        axis=1,
    )
    radius = 0.01
    m = mm.point_to_point_metrics(
        positions, targets, dt, radius, activations=np.full((len(t), 4), 0.5)
    )
    distance = np.linalg.norm((targets - positions).reshape(len(t), -1), axis=-1)
    assert m["success"] == 1.0 and m["final_error"] == pytest.approx(0.0, abs=1e-12)
    assert m["time_to_target"] == pytest.approx(
        np.flatnonzero(distance < radius)[0] * dt
    )
    assert m["target_entries"] == 1.0
    assert m["movement_time"] == pytest.approx(
        duration * (1 - 2 * _TAU_ONSET), abs=1e-3
    )
    assert m["peak_speed"] == pytest.approx(1.875 * 0.175 / duration, rel=1e-3)
    assert m["time_to_peak_ratio"] == pytest.approx(0.5, abs=dt / duration)
    assert m["speed_peaks"] == 1.0
    assert m["straightness"] == pytest.approx(1.0, abs=1e-9)
    assert m["effort"] == pytest.approx(0.25)
    # The 5% segment excludes the smoothest tails, so LDLJ is a little above -5.322.
    assert -5.4 < m["ldlj"] < -4.0
    assert math.isnan(
        mm.point_to_point_metrics(positions, targets, dt, radius)["effort"]
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_point_to_point_metrics_on_very_short_trials(n):
    # Episodes can terminate after a step or two; their smoothness is undefined.
    positions = np.linspace(0.0, 0.01, n)[:, None] * [1.0, 0.0, 0.0]
    m = mm.point_to_point_metrics(positions, [0.1, 0.0, 0.0], 0.02, 0.01)
    assert math.isnan(m["ldlj"])
    assert m["final_error"] == pytest.approx(0.1 - positions[-1, 0])


def test_mean_metrics_skips_undefined_values():
    out = mm.mean_metrics(
        [{"a": 1.0, "b": math.nan}, {"a": 3.0, "b": 2.0}, {"a": 2.0, "b": math.nan}]
    )
    assert out == {"a": 2.0, "b": 2.0}
    assert math.isnan(mm.mean_metrics([{"a": math.nan}])["a"])


# ── Pointing trials and Fitts' law over trials ───────────────────────────────


def test_pointing_trial_metrics_fitts_quantities():
    dt, duration = 0.01, 0.6
    t = np.arange(0.0, 0.8 + 1e-9, dt)
    start, target = np.array([0.1, 0.2, 0.0]), np.array([0.4, 0.6, 0.0])  # D = 0.5
    axis = (target - start) / 0.5
    overshoot = 0.004
    positions = mm.minimum_jerk(start, target + overshoot * axis, duration, t).position
    m = mm.pointing_trial_metrics(positions, target, 0.05, dt, selected=True)
    assert m["success"] == 1.0
    assert m["trial_time"] == pytest.approx((len(t) - 1) * dt)
    assert m["distance"] == pytest.approx(0.5)
    assert m["width"] == pytest.approx(0.1)
    assert m["index_of_difficulty"] == pytest.approx(math.log2(6.0))
    assert m["endpoint_deviation"] == pytest.approx(overshoot)
    assert m["effective_amplitude"] == pytest.approx(0.5 + overshoot)
    assert m["final_error"] == pytest.approx(overshoot)
    missed = mm.pointing_trial_metrics(positions, target, 0.05, dt, selected=False)
    assert missed["success"] == 0.0
    for key in ("trial_time", "effective_amplitude", "endpoint_deviation"):
        assert math.isnan(missed[key])
    assert missed["index_of_difficulty"] == m["index_of_difficulty"]


def test_pointing_trial_metrics_inside_overrides_the_distance_test():
    # E.g. a button: "inside" is a press, not the distance to the centre.
    dt = 0.05
    positions = np.linspace(0.0, 0.3, 9)[:, None] * [1.0, 0.0, 0.0]
    inside = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    m = mm.pointing_trial_metrics(
        positions,
        [0.3, 0.0, 0.0],
        0.01,
        dt,
        selected=True,
        inside=inside,
        dwell_time=4 * dt,
    )
    assert m["time_to_target"] == pytest.approx(2 * dt)
    assert m["time_to_acquire"] == pytest.approx(5 * dt)
    assert m["target_entries"] == 2.0
    with pytest.raises(ValueError):
        mm.pointing_trial_metrics(
            positions, [0.3, 0.0, 0.0], 0.01, dt, selected=True, inside=inside[:-1]
        )


def _pointing_trials(starts, targets, endpoints, width, movement_times):
    """Selected two-sample trials (start, endpoint) with given movement times."""
    trials = []
    for s, g, e, mt in zip(starts, targets, endpoints, movement_times):
        trial = mm.pointing_trial_metrics(
            np.stack([s, e]), g, width / 2, 0.1, selected=True
        )
        trials.append({**trial, "trial_time": float(mt)})
    return trials


def _iso_conditions(rng, conditions, n=12):
    """Random multi-directional trials per (D, W) condition, as in ISO 9241-411."""
    trials, labels, expected = [], [], []
    for label, (distance, width) in enumerate(conditions):
        angle = rng.uniform(0.0, 2 * np.pi, n)
        axis = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        starts = rng.normal(0.0, 0.05, (n, 2))
        targets = starts + distance * axis
        endpoints = targets + rng.normal(0.0, width / 4.133, n)[:, None] * axis
        trials += _pointing_trials(starts, targets, endpoints, width, np.ones(n))
        labels += [label] * n
        expected.append(mm.effective_parameters(starts, targets, endpoints))
    return trials, labels, expected


def test_fitts_conditions_match_effective_parameters_for_one_width():
    rng = np.random.default_rng(3)
    trials, labels, expected = _iso_conditions(
        rng, [(0.2, 0.02), (0.3, 0.05), (0.15, 0.06)]
    )
    conditions = mm.fitts_conditions(trials, conditions=labels)
    assert [c.trials for c in conditions] == [12, 12, 12]
    for c, e in zip(conditions, expected):
        assert c.effective_distance == pytest.approx(e.distance, rel=1e-12)
        assert c.effective_width == pytest.approx(e.width, rel=1e-12)
        assert c.effective_index_of_difficulty == pytest.approx(
            e.index_of_difficulty, rel=1e-12
        )
        assert c.effective_width_ratio == pytest.approx(e.width / c.width, rel=1e-12)


def test_fitts_summary_recovers_a_known_effective_law():
    rng = np.random.default_rng(4)
    trials, labels, expected = _iso_conditions(
        rng, [(0.2, 0.02), (0.3, 0.05), (0.15, 0.06), (0.4, 0.01)]
    )
    a, b = 0.15, 0.2
    ide = np.array([e.index_of_difficulty for e in expected])
    for trial, label in zip(trials, labels):
        trial["trial_time"] = a + b * ide[label]
    out = mm.fitts_summary(trials, conditions=labels)
    assert out["fitts_trials"] == 48.0 and out["fitts_conditions"] == 4.0
    assert out["fitts_effective_intercept"] == pytest.approx(a)
    assert out["fitts_effective_slope"] == pytest.approx(b)
    assert out["fitts_effective_r_squared"] == pytest.approx(1.0)
    assert out["fitts_throughput"] == pytest.approx(np.mean(ide / (a + b * ide)))
    assert out["fitts_effective_index_of_difficulty"] == pytest.approx(ide.mean())
    assert out["fitts_index_of_difficulty"] == pytest.approx(
        np.mean([t["index_of_difficulty"] for t in trials])
    )
    assert np.isfinite(out["fitts_slope"])  # nominal regression over the 48 trials


def test_fitts_conditions_pool_one_difficulty_across_scales():
    # Fitts' law depends on D / W only: trials of one D / W at two sizes form one
    # condition whose IDe and We / W equal those of the unscaled trials.
    rng = np.random.default_rng(5)
    trials, _, (expected,) = _iso_conditions(rng, [(0.2, 0.05)], n=20)
    scaled = []
    for i, trial in enumerate(trials):
        c = 3.0 if i % 2 else 1.0
        lengths = ("width", "effective_amplitude", "endpoint_deviation")
        scaled.append({**trial, **{k: c * trial[k] for k in lengths}})
    (pooled,) = mm.fitts_conditions(scaled, conditions=[0] * len(scaled))
    assert pooled.effective_index_of_difficulty == pytest.approx(
        expected.index_of_difficulty, rel=1e-12
    )
    assert pooled.effective_width_ratio == pytest.approx(
        expected.width / 0.05, rel=1e-12
    )


def test_fitts_conditions_bin_random_trials_by_index_of_difficulty():
    rng = np.random.default_rng(6)
    easy, _, _ = _iso_conditions(rng, [(0.1, 0.05)], n=10)  # ID = log2(3)
    hard, _, _ = _iso_conditions(rng, [(0.4, 0.05)], n=10)  # ID = log2(9)
    trials = [t for pair in zip(hard, easy) for t in pair]  # interleaved
    low, high = mm.fitts_conditions(trials, n_bins=2)
    assert low.trials == high.trials == 10
    assert low.index_of_difficulty == pytest.approx(math.log2(3.0))
    assert high.index_of_difficulty == pytest.approx(math.log2(9.0))


def test_fitts_summary_undefined_cases():
    rng = np.random.default_rng(7)
    trials, labels, _ = _iso_conditions(rng, [(0.2, 0.02)], n=2)
    unselected = {**trials[0], "trial_time": math.nan, "endpoint_deviation": math.nan}
    out = mm.fitts_summary(trials + [unselected])
    assert out["fitts_trials"] == 2.0 and out["fitts_conditions"] == 1.0
    assert np.isfinite(out["fitts_throughput"])
    assert math.isnan(out["fitts_slope"]) and math.isnan(out["fitts_effective_slope"])
    empty = mm.fitts_summary([])
    assert empty["fitts_trials"] == 0.0 and math.isnan(empty["fitts_throughput"])
    # Selections behind the start (De < 0, e.g. without movement): IDe is undefined.
    behind = _pointing_trials(
        np.zeros((2, 2)), [[1.0, 0.0]] * 2, [[-0.2, 0.0], [-0.3, 0.0]], 0.1, [1.0, 1.0]
    )
    (condition,) = mm.fitts_conditions(behind)
    assert condition.effective_distance < 0
    assert math.isnan(condition.effective_index_of_difficulty)
    assert mm.fitts_summary(behind)["fitts_conditions"] == 0.0
    with pytest.raises(ValueError):
        mm.fitts_conditions(trials, conditions=labels[:1])
    with pytest.raises(ValueError):
        mm.fitts_conditions(trials, n_bins=0)


def test_min_jerk_planner_uses_the_reference_polynomial():
    n, duration = 51, 0.8
    tau = np.linspace(0.0, 1.0, n)
    p, pd, pdd = _min_jerk_spaces(n, duration)
    np.testing.assert_allclose(p, 10 * tau**3 - 15 * tau**4 + 6 * tau**5, atol=1e-12)
    np.testing.assert_allclose(
        pd, (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration, atol=1e-12
    )
    np.testing.assert_allclose(
        pdd, (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2, atol=1e-12
    )
    start, goal = np.array([0.1, -0.2]), np.array([0.5, 0.3])
    waypoints = generate_joint_space_min_jerk(start, goal, 1.0, 0.1)
    reference = mm.minimum_jerk(start, goal, 1.0, np.linspace(0.0, 1.0, len(waypoints)))
    np.testing.assert_allclose(
        [w["position"] for w in waypoints], reference.position, atol=1e-12
    )
