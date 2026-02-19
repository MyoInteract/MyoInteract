# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from datetime import datetime
import warnings
import pickle
import h5py

import jax

from absl import app
from absl import flags
from absl import logging
from myosuite.train.utils.train import train_or_load_checkpoint
from myosuite.train.utils.wrapper import _maybe_wrap_env_for_evaluation
from myosuite.envs.myo.myouser.evaluate import evaluate_policy
from myosuite.envs.myo.myouser.utils import render_traj, ProgressLogger, set_global_seed
from etils import epath
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media

from tensorboardX import SummaryWriter
import wandb

from mujoco_playground import registry


import hydra
from myosuite.envs.myo.myouser.hydra_cli import Config
from omegaconf import OmegaConf
from ml_collections.config_dict import ConfigDict


os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
try:
  import madrona_mjx
  location = madrona_mjx.__file__.split('/src/')[0]
  os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = f"{location}/build/kernel_cache"
  os.environ["MADRONA_BVH_KERNEL_CACHE"] = f"{location}/build/bvh_cache"
  print(f'Using cached Madrona MJX kernels at: {location}/build')
except:
  print("Madrona MJX not found, can't use any vision components for training!")

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")



def train(cfg: Config):
  container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
  container['env']['vision'] = container['vision']
  config = ConfigDict(container)
  print(f"Config: ", OmegaConf.to_yaml(container), sep='\n')
  """Run training and evaluation for the specified environment."""

  # Set global seed for reproducibility
  set_global_seed(config.run.seed)

  print(f"Current backend: {jax.default_backend()}")
  # Load environment configuration
  env_cfg = config.env

  ppo_params = config.rl

  # Generate unique experiment name
  if not config.wandb.enabled:  
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_cfg.env_name}-{timestamp}"
    suffix = "" if config.run.suffix is None else f"-{config.run.suffix}"
    exp_name += suffix
  else:
    exp_name = config.wandb.name
  print(f"Experiment name: {exp_name}")
  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Set number of training steps to 0 if play_only
  if config.run.play_only:
    config.rl.num_timesteps = 0
    ##TODO: use config.env.eval_mode rather than config.run.play_only (-> config.run.eval_mode?)

  # Initialize Weights & Biases if required
  if config.wandb.enabled and not config.run.play_only:
    wandb_api = wandb.Api()
    wandb_params = config.wandb.to_dict()
    wandb_params.pop('enabled')
    if wandb.run is None:
      print("Initializing new wandb run")
      wandb_run = wandb.init(**wandb_params)
    else:
      wandb_run = wandb.run
      print("Using existing wandb run with id: ", wandb_run.id)
    wandb_run = wandb_api.run(f"{wandb_params['entity']}/{wandb_params['project']}/{wandb_run.id}")
    wandb.config.update(config.to_dict())
    wandb.config.update({"env_name": env_cfg.env_name})
  else:
    wandb_run = None

  # Initialize TensorBoard if required
  if config.run.use_tb and not config.run.play_only:
    writer = SummaryWriter(logdir)
  else:
    writer = None
  
  progress_logger = ProgressLogger(writer=writer, ppo_params=ppo_params, logdir=logdir,
                                   local_plotting=config.run.local_plotting, log_wandb=config.wandb.enabled and not config.run.play_only, log_tb=config.run.use_tb, log_gradio=config.run.using_gradio)
  progress_fn = progress_logger.progress

  ## TRAINING/LOADING CHECKPOINT
  # Train or load the model
  env, make_inference_fn, params = train_or_load_checkpoint(env_cfg.env_name, config,
                    eval_mode=env_cfg.eval_mode,
                    logdir=logdir,
                    checkpoint_path=config.rl.load_checkpoint_path,
                    progress_fn=progress_fn,
                    wandb_run=wandb_run,
                    log_wandb_videos=config.wandb.enabled and not config.run.play_only and config.run.log_wandb_videos,
                    log_wandb_checkpoints=config.wandb.enabled and not config.run.play_only,
                    vision=config.vision.enabled,
                    domain_randomization=config.run.domain_randomization,
                    rscope_envs=config.run.rscope_envs,
                    deterministic_rscope=config.run.deterministic_rscope,
                    seed=config.run.seed)
  if ppo_params.num_timesteps > 0:
    print("Done training.")
  else:
    print("Done loading checkpoint.")

  times = progress_logger.times
  if ppo_params.num_timesteps > 0 and len(times) > 2:
    # print(f'time to JIT compile: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

  ## STORING CHECKPOINT
  with open(logdir / 'playground_params.pickle', 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  ## EVALUATING CHECKPOINT
  print("Starting inference...")

  # Prepare evaluation episodes
  n_episode_runs = 1
    
  env, n_randomizations = _maybe_wrap_env_for_evaluation(eval_env=env, seed=config.run.eval_seed)
  if n_randomizations is not None:
      ## Multiply n_episodes with num of different/randomized episodes required by eval_env
      n_episodes = n_episode_runs * n_randomizations
      logging.info(f"Environment requires a multiple of {n_randomizations} evaluation episodes. Will run {n_episodes} in total.")
  else:
    n_episodes = n_episode_runs * 20
  
  # Jit required functions
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  jit_reset = jax.jit(env.eval_reset)
  jit_step = jax.jit(env.step)

  # Prepare for evaluation
  #TODO: pass rng instead of seed to evaluate_policy, to avoid correlation between rng used for _maybe_wrap_env_for_evaluation and for further evaluation reset/step calls (re-generated using same seed)
  rollouts, log_type = evaluate_policy(#checkpoint_path=_LOAD_CHECKPOINT_PATH.value, env_name=_ENV_NAME.value,
                            eval_env=env, jit_inference_fn=jit_inference_fn, jit_reset=jit_reset, jit_step=jit_step,
                            seed=config.run.eval_seed,
                            n_episodes=n_episodes)  #config.run.eval_episodes) 
  render_every = 2
  fps = 1.0 / env.dt / render_every
  print(f"FPS for rendering: {fps}")
  
  if log_type == "videos":  
    (rollouts, videos) = rollouts
    if videos[0][0].shape[-1] == 1:
      videos = [frame.squeeze() for v in videos for frame in v]
    else:
      videos = [frame for v in videos for frame in v]
    media.write_video(logdir / "madrona_rollout.mp4", videos, fps=fps)
    print("Rollout video saved as 'madrona_rollout.mp4'.")
    if config.wandb.enabled and not config.run.play_only:
      wandb.log({'final_policy/madrona_view': wandb.Video(str(logdir / "madrona_rollout.mp4"), format="mp4")})  #, fps=fps)})

  print(f"Return: {jp.array([r.reward for rollout in rollouts for r in rollout]).sum()}")
  print(f"env: {env_cfg.env_name}")
  print(f"#episodes: {len(rollouts)}")

  # Render and save the rollout
  with h5py.File(logdir / 'traj.h5', 'w') as h5f:
      h5f.create_dataset('qpos', data=[s.data.qpos for rollout in rollouts for s in rollout])
      h5f.create_dataset('ctrl', data=[s.data.ctrl for rollout in rollouts for s in rollout])
      h5f.close()
  with open(logdir / 'traj.pickle', 'wb') as handle:
      pickle.dump(rollouts, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  # scene_option = mujoco.MjvOption()
  # scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  # scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  # render front view
  render_every = 1
  fps = 1.0 / env.dt / render_every
  print(f"FPS for rendering: {fps}")
  frames = render_traj(
      rollouts, env, height=480, width=640, camera="fixed-eye",
      notebook_context=False,
      render_every=render_every,
      #scene_option=scene_option,
  )
  media.write_video(logdir / "rollout.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout.mp4'.")
  if config.wandb.enabled and not config.run.play_only:
    wandb.log({'final_policy/front_view': wandb.Video(str(logdir / "rollout.mp4"), format="mp4")})  #, fps=fps)})

  # render side view
  frames = render_traj(
      rollouts, env, height=480, width=640, camera=None,
      notebook_context=False,
      #scene_option=scene_option
  )
  media.write_video(logdir / "rollout_1.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout_1.mp4'.")
  if config.wandb.enabled and not config.run.play_only:
    wandb.log({'final_policy/side_view': wandb.Video(str(logdir / "rollout_1.mp4"), format="mp4")})  #, fps=fps)})
  


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
  train(cfg)

if __name__ == "__main__":
  jax.config.parse_flags_with_absl()  #allow for debugging flags such as --jax_debug_nans=True or --jax_disable_jit=True
  main()