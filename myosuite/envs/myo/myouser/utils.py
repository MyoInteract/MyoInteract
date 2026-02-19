from typing import Callable, List, Optional, Sequence
import mujoco
from mujoco_playground import State
from mujoco_playground._src import mjx_env
import numpy as np

import mediapy as media
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
import gradio as gr


def render_traj(rollouts: List[List[State]],
                eval_env: mjx_env.MjxEnv, 
                height: int = 240,
                width: int = 320,
                render_every: int = 1, 
                camera: Optional[int | str] = "fixed-eye",
                scene_option: Optional[mujoco.MjvOption] = None,
                modify_scene_fns: Optional[
                    Sequence[Callable[[mujoco.MjvScene], None]]
                ] = None,
                notebook_context: bool = True):
    ## if called outside a jupyter notebook file, use notebook_context=False

    if isinstance(rollouts[0], State):
      # required for backward compatibility with old checkpoints; rollouts[0] should be List[State]
      rollouts_combined = rollouts
    else:
      rollouts_combined = [r for rollout in rollouts for r in rollout]
    traj = rollouts_combined[::render_every]
    if modify_scene_fns is not None:
        modify_scene_fns = modify_scene_fns[::render_every]
    frames = eval_env.render(traj, height=height, width=width, camera=camera,
                            scene_option=scene_option, modify_scene_fns=modify_scene_fns)
    # rewards = [s.reward for s in rollout]
    
    if notebook_context:
        media.show_video(frames, fps=1.0 / eval_env.dt / render_every)
    else:
        #return media.show_video(frames, fps=1.0 / eval_env.dt / render_every, return_html=True)
        return frames

def update_target_visuals(scn, target_pos, target_size,
                          rgba=[0., 1., 0., 1.]):
    """Updates newly created target visuals in the scene.
    Note: Requires to hide any target visuals included in the MjModel XML."""

    assert scn.ngeom < scn.maxgeom, "Too many geoms in the scene"
    scn.ngeom += 1

    # initialise a new sphere
    mujoco.mjv_initGeom(
          scn.geoms[scn.ngeom-1],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[target_size.item(), 0, 0],
          pos=target_pos,
          mat=np.eye(3).flatten(),
          rgba=np.array(rgba).astype(np.float32)
      )
    # mujoco.mjv_connector(scn.geoms[scn.ngeom-1],
    #                     mujoco.mjtGeom.mjGEOM_CAPSULE, target_size.item(),
    #                     target_pos, target_pos + np.array([1e-6, 0, 0]))

class ProgressLogger:
  def __init__(self, writer=None, ppo_params=None, local_plotting=False, logdir=None, log_wandb=True, log_tb=True, log_gradio=False):
    self.times = [datetime.now()]
    self.writer = writer
    self.ppo_params = ppo_params
    self.local_plotting = local_plotting
    self.log_wandb = log_wandb
    self.log_tb = log_tb
    self.log_gradio = log_gradio
    if log_gradio:
      gr.Info(f"Setting up progress logger to log to wandb and show some info updates in Gradio!")

    if self.local_plotting:
      assert logdir is not None, "logdir must be provided if local_plotting is True"
      self.logdir = logdir

      class PlotParams(object):
        def __init__(self, ppo_params):
          self.x_data = []
          self.y_data = []
          self.ydataerr = []
          self.times = [datetime.now()]

          self.x_data_train = []
          self.y_data_train = []
          self.y_data_train_length = []
          self.y_data_train_success = []
          self.y_data_train_curriculum_state = []

          self.max_y, self.min_y = 150, 0
          self.max_episode_length = ppo_params.episode_length
      self._plt_params = PlotParams(self.ppo_params)
  
  def progress(self, num_steps, metrics):
    self.times.append(datetime.now())
    if len(self.times) == 2:
        print(f'time to JIT compile: {self.times[1] - self.times[0]}')
        print(f'Starting training...')
        if self.log_gradio:
          time_diff = f"Starting training! Time to JIT compile: {self.times[1] - self.times[0]}"
          gr.Info(time_diff)

    # Log to Weights & Biases
    rules = [
        ("reward/", ["reward"]),
        ("success/", ["target", "distance", "success"]),
    ]

    def rename_key(key: str) -> str:
        k = key.lower()
        for replacement, triggers in rules:
            if any(t in k for t in triggers):
                return key.replace("episode/", replacement)
        return key

    wandb_metrics = {rename_key(k): v for k, v in metrics.items()}
    if self.log_wandb:
      wandb.log({**wandb_metrics}, step=num_steps)#'num_steps': num_steps, 

    if self.log_gradio:
      gr.Info(f"Logged step {num_steps} to wandb!")


    # Log to TensorBoard
    if self.log_tb and self.writer is not None:
      for key, value in metrics.items():
        self.writer.add_scalar(key, value, num_steps)
      self.writer.flush()

    if 'eval/episode_reward' in metrics:
    # if self.ppo_params.num_evals > 0:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f} episode_length={metrics['eval/avg_episode_length']:.3f}")
    if not self.log_wandb and not self.log_tb and self.ppo_params.log_training_metrics:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )
    
    if self.local_plotting:
      if self.ppo_params.num_evals > 0 and 'eval/episode_reward' in metrics:
        ## called during evaluation
        # print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
        #     task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
        #     episode length: {metrics['eval/avg_episode_length']}")
        self._plt_params.times.append(datetime.now())
        self._plt_params.x_data.append(num_steps)
        self._plt_params.y_data.append(metrics['eval/episode_reward'])
        self._plt_params.ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([self._plt_params.min_y, self._plt_params.max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={self._plt_params.y_data[-1]:.3f}')

        plt.errorbar(
            self._plt_params.x_data, self._plt_params.y_data, yerr=self._plt_params.ydataerr)
        plt.show()
        
        fig_path = self.logdir / 'progress.png'
        plt.savefig(fig_path)
      elif self.ppo_params.log_training_metrics and 'episode/sum_reward' in metrics:
        ## called during training
            # print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
            # task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
            # episode length: {metrics['eval/avg_episode_length']}")
        self._plt_params.times.append(datetime.now())
        self._plt_params.x_data_train.append(num_steps)
        self._plt_params.y_data_train.append(metrics['episode/sum_reward'])
        self._plt_params.y_data_train_length.append(metrics['episode/length'])
        self._plt_params.y_data_train_success.append(metrics['episode/success_rate'])
        self._plt_params.y_data_train_curriculum_state.append(metrics['episode/target_area_dynamic_width_scale'])

        ## Reward
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([self._plt_params.min_y, self._plt_params.max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={self._plt_params.y_data_train[-1]:.3f}')

        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train) # yerr=ydataerr)
        plt.show()

        fig_path = self.logdir / 'progress_train_reward.png'
        plt.savefig(fig_path)
        plt.close()

        ## Episode length
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([0, self._plt_params.max_episode_length * 1.1])

        plt.xlabel('# environment steps')
        plt.ylabel('episode length')
        plt.title(f'length={self._plt_params.y_data_train_length[-1]:.3f}')
        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train_length) # yerr=ydataerr)
        plt.show()
        plt.show()

        fig_path = self.logdir / 'progress_train_length.png'
        plt.savefig(fig_path)
        plt.close()
        
        ## Success rate
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([0, 1])

        plt.xlabel('# environment steps')
        plt.ylabel('success rate')
        plt.title(f'success={self._plt_params.y_data_train_success[-1]:.3f}')
        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train_success) # yerr=ydataerr)
        plt.show()

        fig_path = self.logdir / 'progress_train_success.png'
        plt.savefig(fig_path)
        plt.close()
        
        ## Curriculum state
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([0, 1])

        plt.xlabel('# environment steps')
        plt.ylabel('curriculum state')
        plt.title(f'width={self._plt_params.y_data_train_curriculum_state[-1]:.3f}')
        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train_curriculum_state) # yerr=ydataerr)
        plt.show()

        fig_path = self.logdir / 'progress_train_curriculum.png'
        try:
          plt.savefig(fig_path)
          plt.close()
        except:
          pass

def set_global_seed(seed=0):
    """Set global random seeds for reproducible results."""
    import random
    import numpy as np
        
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    print(f"Global random seed set to {seed} for reproducible results")