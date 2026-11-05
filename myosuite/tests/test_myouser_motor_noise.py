"""Motor-noise tests for MyoUserBase.get_ctrl: per-muscle draws and red noise carried in state.info."""

import os
import tempfile
import unittest

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground import State

from myosuite.envs.myo.myouser.base import MyoUserBase

N_MUSCLES = 6
N_DRAWS = 20000

_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="slide" type="slide" axis="1 0 0" range="-0.1 0.1"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <tendon>{tendons}</tendon>
  <actuator>{muscles}</actuator>
</mujoco>
"""


class _NoiseEnv(MyoUserBase):
    """Minimal MyoUserBase that follows MyoUserUniversal's reset/step info conventions (no physics)."""

    def reset(self, rng, **kwargs):
        data = self._reset_bm_model(rng)
        info = {"rng": rng, **self._reset_motor_noise()}
        return State(data, jp.zeros(1), jp.zeros(()), jp.zeros(()), {}, info)

    def step(self, state, action):
        rng, rng_ctrl = jax.random.split(state.info["rng"])
        new_ctrl, noise_info = self.get_ctrl(state, action, rng_ctrl)
        state.info.update(noise_info, rng=rng)
        return state.replace(data=state.data.replace(ctrl=new_ctrl, time=state.data.time + self.dt))

    def auto_reset(self, state_before_reset):
        return self.reset(state_before_reset.info["rng"])


def _make_env(**noise_params):
    params = dict(sigdepnoise_type=None, sigdepnoise_level=0.103, constantnoise_type=None, constantnoise_level=0.185)
    params.update(noise_params)
    xml = _XML.format(
        tendons="".join(f'<fixed name="t{i}"><joint joint="slide" coef="1"/></fixed>' for i in range(N_MUSCLES)),
        muscles="".join(f'<muscle name="m{i}" tendon="t{i}"/>' for i in range(N_MUSCLES)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "muscles.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)
        config = config_dict.create(
            model_path=path, ctrl_dt=0.05, sim_dt=0.002, eval_mode=False,
            vision=config_dict.create(enabled=False),
            muscle_config=config_dict.create(muscle_condition=None, sex=None, control_type="default",
                                             noise_params=config_dict.create(**params)),
            task_config=config_dict.create(max_trials=1, reset_type="zero"),
        )
        return _NoiseEnv(config)


def _white_noise(env, u):
    """Applied muscle noise for N_DRAWS independent keys (jit + vmap), shape (N_DRAWS, N_MUSCLES)."""
    state = env.reset(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), N_DRAWS)
    ctrl = jax.jit(jax.vmap(lambda k: env.get_ctrl(state, u, k)[0]))(keys)
    return np.asarray(ctrl) - np.asarray(u)


def _red_rollout(env, n_steps, n_envs=2000, auto_reset_at=None):
    """SD of the applied noise after each jitted step; optionally auto-reset `data` but keep `info`."""
    u = jp.full(N_MUSCLES, 0.5)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, None)))
    states = reset(jax.random.split(jax.random.PRNGKey(0), n_envs))
    sds = []
    for t in range(n_steps):
        if t == auto_reset_at:
            # As BetterAutoResetWrapper / playground's AutoResetWrapper do: fresh data, pre-reset info.
            states = states.replace(data=reset(jax.random.split(jax.random.PRNGKey(2), n_envs)).data)
        states = step(states, u)
        sds.append(float(np.std(np.asarray(states.data.ctrl) - 0.5)))
    return np.array(sds), states


def _max_offdiag_corr(x):
    c = np.corrcoef(x, rowvar=False)
    return np.abs(c[~np.eye(c.shape[0], dtype=bool)]).max()


class TestMotorNoise(unittest.TestCase):

    def test_noise_off_leaves_controls_unchanged(self):
        env = _make_env()
        action = jp.linspace(-0.5, 1.5, N_MUSCLES)
        ctrl, noise_info = jax.jit(env.get_ctrl)(env.reset(jax.random.PRNGKey(0)), action, jax.random.PRNGKey(1))
        np.testing.assert_array_equal(ctrl, np.clip(action, 0, 1))
        self.assertEqual(noise_info, {})

    def test_white_noise_is_independent_across_muscles(self):
        u = jp.full(N_MUSCLES, 0.5)
        for params in ({"sigdepnoise_type": "white"}, {"constantnoise_type": "white", "constantnoise_level": 0.05}):
            with self.subTest(**params):
                self.assertLess(_max_offdiag_corr(_white_noise(_make_env(**params), u)), 0.05)

    def test_noise_sd(self):
        # Signal-dependent: SD = level * u per muscle; constant: SD = level.
        u = jp.linspace(0.1, 0.6, N_MUSCLES)
        sd = _white_noise(_make_env(sigdepnoise_type="white"), u).std(0)
        np.testing.assert_allclose(sd / (0.103 * np.asarray(u)), 1.0, atol=0.05)
        sd = _white_noise(_make_env(constantnoise_type="white", constantnoise_level=0.05), jp.full(N_MUSCLES, 0.5)).std(0)
        np.testing.assert_allclose(sd / 0.05, 1.0, atol=0.05)

    def test_red_noise_accumulates_across_jitted_steps_and_resets(self):
        for params, step_sd in (({"sigdepnoise_type": "red", "sigdepnoise_level": 0.01}, 0.01 * 0.5),
                                ({"constantnoise_type": "red", "constantnoise_level": 0.01}, 0.01)):
            with self.subTest(**params):
                env = _make_env(**params)
                np.testing.assert_array_equal(env.reset(jax.random.PRNGKey(0)).info["sigdepnoise_acc"], np.zeros(N_MUSCLES))
                sds, states = _red_rollout(env, 30, auto_reset_at=20)
                # Random walk: SD grows with sqrt(t), and restarts after the (auto-)reset.
                np.testing.assert_allclose(sds[[0, 19, 20, 29]] / (step_sd * np.sqrt([1, 20, 1, 10])), 1.0, atol=0.1)
                key = "sigdepnoise_acc" if "sigdepnoise_type" in params else "constantnoise_acc"
                self.assertEqual(states.info[key].shape, (2000, N_MUSCLES))
                # The same env object can be traced again (no leaked tracer on self).
                _red_rollout(env, 2, n_envs=8)


if __name__ == "__main__":
    unittest.main()
