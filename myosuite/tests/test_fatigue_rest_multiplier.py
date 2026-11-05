"""3CC-r recovery multiplier: r applies only at rest (Rakshit et al. 2021, Eq. 7)."""

import numpy as np
import pytest

from myosuite.envs.myo.fatigue import MUSCLE_FATIGUE_PARAMS, CumulativeFatigue

F, R, r, DT = 0.01, 0.001, 150.0, 0.02


def _fatigue(ma, mr, mf):
    """One-muscle CumulativeFatigue without a MuJoCo model."""
    cf = CumulativeFatigue.__new__(CumulativeFatigue)
    cf._dt, cf.na = DT, 1
    cf._tauact, cf._taudeact = np.array([0.01]), np.array([0.04])
    cf._MA, cf._MR, cf._MF = np.array([ma]), np.array([mr]), np.array([mf])
    cf.TL = np.zeros(1)
    cf._F, cf._R, cf._r = np.array([F]), np.array([R]), np.array([r])
    return cf


def _applied_recovery_rate(ma, mr, mf, tl):
    """Recovery rate rR backed out of one MF update: dMF = (F*MA - rR*MF)*dt."""
    cf = _fatigue(ma, mr, mf)
    cf.compute_act(np.array([tl]))
    return (F * ma - (cf._MF[0] - mf) / DT) / mf


@pytest.mark.parametrize(
    "ma, mr, mf, tl, expected",
    [
        (0.5, 0.3, 0.2, 0.2, R),  # relaxing under load (MA >= TL > 0): no rest multiplier
        (0.1, 0.7, 0.2, 0.6, R),  # ramping up under load
        (0.5, 0.3, 0.2, 0.0, r * R),  # rest
        (0.5, 0.3, 0.2, -0.5, r * R),  # command below the muscle's zero is rest too
    ],
)
def test_rest_multiplier_only_at_rest(ma, mr, mf, tl, expected):
    assert _applied_recovery_rate(ma, mr, mf, tl) == pytest.approx(expected, rel=1e-9)


def test_shoulder_row_is_the_published_shoulder_fit():
    # Frey-Law, Looft & Heitsman 2012, Table 1 (it was a copy of the Knee row)
    shoulder = MUSCLE_FATIGUE_PARAMS["Shoulder"]
    assert (shoulder["F"], shoulder["R"]) == (0.01820, 0.00168)
    assert shoulder != MUSCLE_FATIGUE_PARAMS["Knee"]
