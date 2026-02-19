import os
import functools
import json
from etils import epath
from absl import logging
import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground import registry
import wandb
import mediapy as media

from orbax import checkpoint as ocp
from flax.training import orbax_utils
import matplotlib.pyplot as plt
from collections import defaultdict


# from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.agents.ppo import networks_vision as ppo_networks_vision
# from brax.training.agents.ppo import train as ppo

from myosuite.train.myouser.custom_ppo import train as ppo
from myosuite.train.myouser.custom_ppo import networks_vision_unified as networks
from myosuite.train.utils.wrapper import wrap_myosuite_training, _maybe_wrap_env_for_evaluation
from myosuite.envs.myo.myouser.evaluate import evaluate_policy
from myosuite.envs.myo.myouser.utils import render_traj

# class ProgressEvalVideoLoggerDEPRECATED:
#   def __init__(self, logdir, eval_env,
#                seed=123,
#                deterministic=True,
#                n_episodes=10,
#                ep_length=None,
#                height=480,
#                width=640,
#                cameras=[None],
#                ):
#     self.logdir = logdir
#     ckpt_path = logdir / "checkpoints"
#     ckpt_path.mkdir(parents=True, exist_ok=True)
#     self.checkpoint_path = ckpt_path
#     self.eval_env = eval_env
#     self.seed = seed
#     self.eval_key = jax.random.PRNGKey(seed)
#     self.deterministic = deterministic
#     self.n_episodes = n_episodes

#     if ep_length is None:
#         ep_length = int(eval_env._config.task_config.max_duration / eval_env._config.ctrl_dt)
#     self.episode_length = ep_length

#     self.height = height
#     self.width = width
#     self.cameras = cameras
#     self.n_cameras = len(cameras)

#   def rollout_traj_frames(self, make_policy, params):
#     rollout = []
#     # TODO: use jax.lax.scan instead of for-loop
#     for ep_id in range(self.n_episodes):
#       traj_frames = self._rollout(make_policy, params)
#       rollout.extend(traj_frames)
#     return rollout

#   def _rollout(self, make_policy, params):
#     ## code is partially based on rscope.BraxRolloutSaver class

#     key_unroll, key_reset = jax.random.split(self.eval_key)
#     policy = make_policy(params, deterministic=self.deterministic)
#     state = self.eval_env.reset(key_reset)

#     # Collect complete rollout
#     def step_fn(c):
#       state, key, i, rollout_frames = c
#       key, key_ctrl = jax.random.split(key)
#       ctrl, _ = policy(state.obs, key_ctrl)
#       state = self.eval_env.step(state, ctrl)
#       for camera_id, camera in enumerate(self.cameras):
#         frames = self.eval_env.render_array(self.eval_env.mj_model, state, height=self.height, width=self.width, camera=camera)
#         rollout_frames = rollout_frames.at[camera_id, i].set(frames)
#       # return (state, key), jax.tree.map(lambda x: x[: self.rscope_envs], full_ret)
#       return (state, key, i+1, rollout_frames)
    
#     def cond_fn(c):
#       state, key, i, rollout_frames = c
#       return ~state.done.all()
    
#     # _, rollout = jax.lax.scan(
#     #     step_fn,
#     #     (state, key_unroll),
#     #     None,
#     #     length=self.episode_length,
#     # )
    
#     rollout_frames = jp.zeros((self.n_cameras, self.episode_length, self.height, self.width, 3))  #TODO: allow for different render modes, e.g. RGB-D?
#     state, key, num_steps, rollout_frames = jax.lax.while_loop(
#         cond_fn,
#         step_fn,
#         (state, key_unroll, 0, rollout_frames),
#     )

#     return rollout_frames

#   def progress_eval_video(self, current_step, make_policy, params):
#     raise ValueError  #TODO: this attempt to run jitted rendering does not work, as we can neither store all states in jax arrays for later rendering, nor render directly on traced arrays
#     # Log to Weights & Biases

#     # Rollout trajectory
#     # rollout = evaluate_policy(checkpoint_path=self.checkpoint_path, env_name=env_name)
#     frames = self.rollout_traj_frames(make_policy=make_policy, params=params)
    
#     fps = 1.0 / self.eval_env.dt
#     for camera_id, camera in enumerate(self.cameras):
#         camera_frames = frames[camera_id]
#         if camera is None:
#             camera_suffix = ""
#         else:
#             camera_suffix = "_" + camera
#         media.write_video(self.checkpoint_path / f"{current_step}{camera_suffix}.mp4", camera_frames, fps=fps)
#         wandb.log({f'eval_vis/camera{camera_suffix}': wandb.Video(str(self.checkpoint_path / f"{current_step}{camera_suffix}.mp4"), format="mp4")}, step=current_step)  #, fps=fps)}, step=num_steps)

def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]

def plot_eval_metrics(results, step):
    all_sizes, all_successes = [], []
    for entry in results:
        sizes = ensure_list(entry["target_size"])
        successes = ensure_list(entry["success"])
        for s, succ in zip(sizes, successes):
            s = float(s) if hasattr(s, "item") else s
            all_sizes.append(s)
            all_successes.append(int(succ))

    # Plot Target size vs Success Rate
    fig1 = plt.figure(figsize=(8,5))
    plt.plot(all_sizes, all_successes, marker='o', linestyle='')
    plt.xlabel("Target size")
    plt.ylabel("Success rate")
    plt.title("Success rate vs Target size")
    wandb.log({"success/success_per_target_size": wandb.Image(fig1)}, step=step)
    plt.close(fig1)

    # Plot 2: 3D target positions
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for entry in results:
        positions = ensure_list(entry["target_pos"])
        successes = ensure_list(entry["success"])
        for pos, succ in zip(positions, successes):
            pos = pos.tolist() if hasattr(pos, "tolist") else pos
            pos[1] = -pos[1]
            ax.scatter(*pos, color="green" if succ else "red", s=50)

    ax.set_xlabel('depth'); ax.set_ylabel('left-right'); ax.set_zlabel('height')
    ax.set_title("Target positions (green=success, red=failure)")
    ax.set_xlim(ax.get_xlim()[::-1])
    wandb.log({"success/target_positions": wandb.Image(fig)}, step=step)
    plt.close()

class ProgressEvalLogger:
  def __init__(self, logdir, eval_env,
               seed=123,
               deterministic=True,
               log_wandb_checkpoints=False,
               n_episode_runs=10,
               n_episodes_video=10,
               ep_length=None,
               eval_metrics_keys={},
               height=480,
               width=640,
               cameras=[None],
               ):
    self.logdir = logdir
    if logdir is not None:
        ckpt_path = logdir / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = ckpt_path
        plots_path = logdir / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        self.plots_path = plots_path

    self.seed = seed
    # eval_key = jax.random.PRNGKey(seed)
    # self.eval_key, reset_prepare_keys = jax.random.split(eval_key)
    self.deterministic = deterministic

    self.log_wandb_checkpoints = log_wandb_checkpoints

    # Prepare evaluation episodes
    self.eval_env, self.n_randomizations = _maybe_wrap_env_for_evaluation(eval_env=eval_env, seed=seed)
    if self.n_randomizations is not None:
        ## Multiply n_episodes with num of different/randomized episodes required by eval_env
        n_episodes = n_episode_runs * self.n_randomizations
        logging.info(f"Environment requires a multiple of {self.n_randomizations} evaluation episodes. Will run {n_episodes} in total.")
    else:
       n_episodes = n_episode_runs
    self.n_episodes = n_episodes

    self.n_episodes_video = n_episodes_video

    if ep_length is None:
        ep_length = int(self.eval_env._config.task_config.max_duration / self.eval_env._config.ctrl_dt)
    self.episode_length = ep_length

    # Keys to be logged
    self.eval_metrics_keys = eval_metrics_keys

    self.height = height
    self.width = width
    self.cameras = cameras
    self.n_cameras = len(cameras)

    self.jit_reset = jax.jit(self.eval_env.eval_reset)
    self.jit_step = jax.jit(self.eval_env.step)

  def progress_eval_run(self, current_step, make_policy, params,
                        training_metrics={}):
    
    # store checkpoint locally
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = self.checkpoint_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    # upload checkpoint to wandb
    if self.log_wandb_checkpoints:
        exp_name = self.logdir.name
        artifact = wandb.Artifact(
            name=f'{exp_name}-checkpoints',  
            type="model"
        )
        artifact.add_dir(str(self.checkpoint_path))
        artifact.save()
    
    # Rollout trajectory
    if self.n_episodes > 0:
        jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
        rollouts = evaluate_policy(eval_env=self.eval_env, jit_inference_fn=jit_inference_fn, jit_reset=self.jit_reset, jit_step=self.jit_step, seed=self.seed, n_episodes=self.n_episodes_video)[0]
        results = self.eval_env.eval_metrics(rollouts) #, eval_metrics_keys=self.eval_metrics_keys)
        plot_eval_metrics(results, step=current_step)
        rollout_metrics_keys = rollouts[0][0].metrics.keys()
        rollout_metrics = {f'eval/{k}': jp.mean(jp.array([jp.sum(jp.array([r.metrics[k] for r in rollout])) for rollout in rollouts])) for k in rollout_metrics_keys}
    
    # Calculate metrics
    # final_eval_state = rollout[-1]
    # eval_metrics = final_eval_state.info['eval_metrics']
    # metrics = {}
    # for fn in [np.mean]:  #, np.std]:
    #   suffix = '_std' if fn == np.std else ''
    #   metrics.update({
    #       f'eval/episode_{name}{suffix}': (
    #           fn(value)
    #       )
    #       for name, value in eval_metrics.episode_metrics.items()
    #   })
    # metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    # metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    # # run more rollouts that only store information relevant for computing task metrics
    # (movement_times, rollout_states) = evaluate_policy(eval_env=self.eval_env, jit_inference_fn=jit_inference_fn, jit_reset=self.jit_reset, jit_step=self.jit_step, seed=self.seed, n_episodes=self.n_episodes, log_MT_only=True)[0]
    # task_metrics = self.eval_env.calculate_metrics(movement_times=movement_times, rollout_states=rollout_states) #, eval_metrics_keys=self.eval_metrics_keys)
    ### NOTE: moved to checkpoint callback function, which is called more frequently, such that this videologger callback is reserved for "full" rollouts
    task_metrics = {}

    # Create video that can be uploaded to Weights & Biases
    video_metrics = {}#self.progress_eval_video(rollouts, current_step)

    metrics = {**rollout_metrics, **task_metrics, **video_metrics, **training_metrics}    
    wandb.log(metrics, step=current_step)
    print({**rollout_metrics, **task_metrics})

    return metrics
  

  def progress_eval_video(self, rollouts, current_step):
    # Create video that can be uploaded to Weights & Biases

    fps = 1.0 / self.eval_env.dt

    video_metrics = {}
    for camera in self.cameras:
        if camera is None:
            camera_suffix = ""
        else:
            camera_suffix = "_" + camera
        frames = render_traj(
            rollouts, self.eval_env, height=480, width=640, camera=camera,
            notebook_context=False,
        )
        media.write_video(self.checkpoint_path / f"{current_step}{camera_suffix}.mp4", frames, fps=fps)
        video_metrics[f'eval_vis/camera{camera_suffix}'] = wandb.Video(str(self.checkpoint_path / f"{current_step}{camera_suffix}.mp4"), format="mp4")

    return video_metrics
    # # render front view
    # frames = render_traj(
    #     rollout, self.eval_env, height=480, width=640, camera="fixed-eye",
    #     notebook_context=False,
    # )
    # media.write_video(self.checkpoint_path / f"{current_step}.mp4", frames, fps=fps)
    # wandb.log({'eval_vis/front_view': wandb.Video(str(self.checkpoint_path / f"{current_step}.mp4"), format="mp4")}, step=current_step)  #, fps=fps)}, step=num_steps)

    # # render side view
    # frames = render_traj(
    #     rollout, self.eval_env, height=480, width=640, camera=None,
    #     notebook_context=False,
    # )
    # media.write_video(self.checkpoint_path / f"{num_steps}_1.mp4", frames, fps=fps)
    # wandb.log({'eval_vis/side_view': wandb.Video(str(self.checkpoint_path / f"{num_steps}_1.mp4"), format="mp4")}, step=num_steps)  #, fps=fps)}, step=num_steps)


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )

def train_or_load_checkpoint(env_name, 
                    config,
                    eval_mode=False,
                    logdir=None,
                    checkpoint_path=None,
                    policy_params_fn_checkpoints=None,
                    progress_fn=lambda *args: None,
                    wandb_run=None,
                    log_wandb_videos=False,
                    log_wandb_checkpoints=False,
                    vision=False,
                    domain_randomization=False,
                    rscope_envs=None,
                    deterministic_rscope=True,
                    seed=1,):
    
    # env_cfg = registry.get_default_config(env_name)  #default_config()
    env_cfg = config.env
    ppo_params = config.rl
    if eval_mode:
        ppo_params.num_timesteps = 0  #only load the model, do not train

    ## TODO: fix this properly
    if checkpoint_path == "None":
        checkpoint_path = None
    if env_cfg.muscle_config.noise_params.sigdepnoise_type == "None":
        env_cfg.muscle_config.noise_params.sigdepnoise_type = None
    if env_cfg.muscle_config.noise_params.constantnoise_type == "None":
        env_cfg.muscle_config.noise_params.constantnoise_type = None

    # Handle checkpoint loading
    if checkpoint_path is not None:
        # Convert to absolute path
        ckpt_path = epath.Path(checkpoint_path).resolve()
        if ckpt_path.is_dir() and ckpt_path.name == "checkpoints":
            latest_ckpts = list(ckpt_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = latest_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = ckpt_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None

    # Set up checkpoint directory
    if logdir is not None:
        ckpt_path = logdir / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint path: {ckpt_path}")
    else:
        assert checkpoint_path is not None, "Either logdir or checkpoint_path must be provided"
        print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(config.to_dict(), fp, indent=4)

    if vision:
        env_cfg.vision_mode = vision
        env_cfg.vision.render_batch_size = ppo_params.num_envs
        env_cfg.vision.num_worlds = ppo_params.num_envs
    env = registry.load(env_name, config=env_cfg)
    

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    # network_fn = (
    #     ppo_networks_vision.make_ppo_networks_vision
    #     if vision
    #     else ppo_networks.make_ppo_networks
    # )
    # if hasattr(ppo_params, "network_factory"):
    #     network_factory = functools.partial(
    #         network_fn, **ppo_params.network_factory
    #     )
    # else:
    #     network_factory = network_fn

    network_factory = functools.partial(networks.custom_network_factory, vision=vision, **getattr(ppo_params, "network_factory", {}))

    if domain_randomization:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            env_name
        )

    if vision:
        env = wrap_myosuite_training(
            env,
            vision=True,
            num_vision_envs=env_cfg.vision.render_batch_size,
            episode_length=ppo_params.episode_length,
            action_repeat=ppo_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )
    else:
        env = wrap_myosuite_training(
            env,
            episode_length=ppo_params.episode_length,
            action_repeat=ppo_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )

    num_eval_envs = (
        ppo_params.num_envs
        if vision
        else ppo_params.get("num_eval_envs", 128)
    )

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]
    
    if "load_checkpoint_path" in training_params:
        del training_params["load_checkpoint_path"]

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        # policy_params_fn=policy_params_fn,
        seed=seed,
        restore_checkpoint_path=restore_checkpoint_path,
        # save_checkpoint_path=ckpt_path,
        # wrap_env_fn=None if vision else wrapper.wrap_for_brax_training,
        # wrap_env_fn=(lambda x, **kwargs: x) if vision else wrap_myosuite_training,
        wrap_env_fn=(lambda x, **kwargs: x),
        num_eval_envs=num_eval_envs,
    )

    # Load evaluation environment
    eval_env = (
        None if vision else registry.load(env_name, config=env_cfg)
    )    
    # eval_env = env
    progress_eval_logger = None
    if rscope_envs:
        # Interactive visualisation of policy checkpoints
        from rscope import brax as rscope_utils

        if not vision:
            rscope_env = registry.load(env_name, config=env_cfg)
            rscope_env = wrap_myosuite_training(  #wrapper.wrap_for_brax_training(
                rscope_env,
                episode_length=ppo_params.episode_length,
                action_repeat=ppo_params.action_repeat,
                randomization_fn=training_params.get("randomization_fn"),
            )
        else:
            rscope_env = env
        if not hasattr(rscope_env, "model_assets"):
            assets = {}
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            _XML_PATH = epath.Path(os.path.join(curr_dir, "../../../simhive/uitb_sim/assets"))
            for f in _XML_PATH.glob("*"):
                if f.is_file() and f.name.endswith(".stl"):
                    assets[f"../../../../simhive/uitb_sim/assets/{f.name}"] = f.read_bytes()
            rscope_env.model_assets = assets

        rscope_handle = rscope_utils.BraxRolloutSaver(
            rscope_env,
            ppo_params,
            vision,
            rscope_envs,
            deterministic_rscope,
            jax.random.PRNGKey(seed),
            rscope_fn,
        )

        def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
            # progress_fn_eval(current_step, make_policy, params)
            rscope_handle.set_make_policy(make_policy)
            rscope_handle.dump_rollout(params)
    else:
        if log_wandb_videos:
            progress_eval_logger = ProgressEvalLogger(logdir=logdir, eval_env=eval_env,
                                                                seed=config.run.eval_seed,
                                                                log_wandb_checkpoints=log_wandb_checkpoints,
                                                                n_episode_runs=10,
                                                                n_episodes_video=10,
                                                                #ep_length=80,
                                                                cameras=["fixed-eye", None]
                                                                )
            policy_params_fn = progress_eval_logger.progress_eval_run
        else:
            policy_params_fn = lambda *args: None
    
    if policy_params_fn_checkpoints is None:
        if not eval_mode:
            if progress_eval_logger is None:
                progress_eval_logger = ProgressEvalLogger(logdir=logdir, eval_env=eval_env,
                                                                    seed=config.run.eval_seed,
                                                                    log_wandb_checkpoints=log_wandb_checkpoints,
                                                                    n_episode_runs=10,
                                                                    n_episodes_video=10,
                                                                    #ep_length=80,
                                                                    cameras=["fixed-eye", None]
                                                                    )
            policy_params_fn_checkpoints = progress_eval_logger.progress_eval_run
        else:
            policy_params_fn_checkpoints = lambda *args: None


        # # Define policy parameters function for saving checkpoints
        # def policy_params_fn_checkpoints(current_step, make_policy, params):
        #     """Function to save policy parameters."""
        #     orbax_checkpointer = ocp.PyTreeCheckpointer()
        #     save_args = orbax_utils.save_args_from_target(params)
        #     path = ckpt_path / f"{current_step}"
        #     orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        #     if log_wandb_checkpoints:
        #         exp_name = logdir.name
        #         artifact = wandb.Artifact(
        #             name=f'{exp_name}-checkpoints',  
        #             type="model"
        #         )
        #         artifact.add_dir(str(ckpt_path))
        #         artifact.save()
            
        #     # do short rollouts and calculate SL metrics
        #     (movement_times, rollout_states) = evaluate_policy(eval_env=eval_env, jit_inference_fn=jit_inference_fn, jit_reset=self.jit_reset, jit_step=self.jit_step, seed=self.seed, n_episodes=self.n_episodes)[0]
        #     task_metrics = eval_env.calculate_metrics(movement_times=movement_times, rollout_states=rollout_states) #, eval_metrics_keys=self.eval_metrics_keys)
        #     wandb.log(task_metrics, step=current_step)

    if not eval_mode:
        print("Starting to JIT compile...")
        
    # Train or load the model
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,  #lambda *args: None,
        policy_params_fn_checkpoints=policy_params_fn_checkpoints,
        eval_env=eval_env,
    )
    if vision:
        eval_env = env
    eval_env.enable_eval_mode()

    # Make sure all checkpoints and eval videos are uploaded into wandb artifact
    if log_wandb_checkpoints:
        
        exp_name = logdir.name
        artifact = wandb.Artifact(
            name=f'{exp_name}-checkpoints',  
            type="model"
        )
        artifact.add_dir(str(ckpt_path))
        artifact.save()
        
        # delete previous artifacts (need to create multiple as we cannot override existing ones during training)
        for artifact in wandb_run.logged_artifacts():
            if any([artifact.name.split("v")[-1] == str(_i) for _i in range(ppo_params.num_checkpoints)]):
                try:
                    artifact.delete()
                except:
                    pass

    return eval_env, make_inference_fn, params