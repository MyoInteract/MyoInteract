import os, re, glob
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
# Internal loading of video libraries.

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_logdir(filepath):
    """Get latest subdir in logging filepath that was created by Unity App."""
    
    filepath = os.path.expanduser(filepath)
    
    subdirs = natural_sort(os.listdir(filepath))
    
    # only consider directories that contain a non-empty states.csv file, excluding hidden directories (e.g. '.ipynb_checkpoints')
    subdirs = [i for i in subdirs if os.path.isdir(os.path.join(os.path.expanduser(filepath), i)) and ((not os.path.isfile(os.path.join(os.path.expanduser(filepath), i, "states.csv"))) or (os.path.getsize(os.path.join(os.path.expanduser(filepath), i, "states.csv")) > 0)) and (not i.startswith('.')) and (len(os.listdir(os.path.join(filepath, i))) > 0)]
    
    if len(subdirs) > 0:
        last_subdir = subdirs[-1]
        filepath_new = os.path.join(filepath, last_subdir)
        if os.path.isdir(filepath_new):
            return get_logdir(filepath_new)
    
    return filepath

# # copy logged pickles to logging dir
# _SIMULATION_USER_ID = 91  #fake simulation user ID used for testing

# _evaluation_dir = f"~/uitb-sim2vr/user-in-the-box-private/{_TASK_CONDITION}/evaluate/"
# _latest_logdir_condition = get_logdir(os.path.join(_evaluation_dir, "logging"))
# # _latest_logdir = os.path.dirname(os.path.expanduser(_latest_logdir_condition))
# os.popen(f'cp {_evaluation_dir}/*_log.pickle {_latest_logdir_condition}') #copy uitb/MuJoCo logs (pickle files) to recent unity logdir (containing the csv files)
# if len(glob.glob(os.path.expanduser(f'{_evaluation_dir}/*.mp4'))) > 0:
#     os.popen(f'ln -sf {_evaluation_dir}/*.mp4 {_latest_logdir_condition}') #link generated video to recent unity logdir
# os.popen(f'mkdir -p ~/uitb-sim2vr/user-in-the-box-private/datasets/vr-uitb-experiment/{_SIMULATION_USER_ID}/')
# os.popen(f'cp -r {_latest_logdir_condition} ~/uitb-sim2vr/user-in-the-box-private/datasets/vr-uitb-experiment/{_SIMULATION_USER_ID}/') #copy (unity) logdir to datasets

def copy_to_dataset(TASK_CONDITION, SIMULATION_USER_ID):
    # copy logged pickles to logging dir
    _evaluation_dir = f"~/uitb-sim2vr/user-in-the-box-private/{TASK_CONDITION}/evaluate/"
    _latest_logdir_condition = get_logdir(os.path.join(_evaluation_dir, "logging"))
    # _latest_logdir = os.path.dirname(os.path.expanduser(_latest_logdir_condition))
    os.popen(f'cp {_evaluation_dir}/*_log.pickle {_latest_logdir_condition}') #copy uitb/MuJoCo logs (pickle files) to recent unity logdir (containing the csv files)
    if len(glob.glob(os.path.expanduser(f'{_evaluation_dir}/*.mp4'))) > 0:
        os.popen(f'ln -sf {_evaluation_dir}/*.mp4 {_latest_logdir_condition}') #link generated video to recent unity logdir
    os.popen(f'mkdir -p ~/uitb-sim2vr/user-in-the-box-private/datasets/vr-uitb-experiment/{SIMULATION_USER_ID}/')
    os.popen(f'cp -r {_latest_logdir_condition} ~/uitb-sim2vr/user-in-the-box-private/datasets/vr-uitb-experiment/{SIMULATION_USER_ID}/') #copy (unity) logdir to datasets

    print(f"NEW [{SIMULATION_USER_ID}]: {_latest_logdir_condition} successfully registered at '~/uitb-sim2vr/user-in-the-box-private/datasets/vr-uitb-experiment/{SIMULATION_USER_ID}/'.")

## Custom video functions

def display_video(frames, framerate=30):
    anim = _create_animation(frames, framerate=framerate)
    return HTML(anim.to_html5_video())

def _create_animation(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    return animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)

def store_video(filename, frames, framerate=30):
    anim = _create_animation(frames, framerate=framerate)
    
    if filename.endswith(".gif"):
        writer = animation.PillowWriter(fps=framerate)
    elif filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
        writer = animation.FFMpegWriter(fps=framerate) 
    
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    anim.save(filename, writer=writer)
    
    print(f"Animation stored at '{filename}'.")

def add_text_to_frame(frame, text, font="dejavu/DejaVuSans.ttf", pos=(20, 20), color=(255, 0, 0), fontsize=12):
    if isinstance(frame, np.ndarray):
        frame = PIL.Image.fromarray(frame)
    
    draw = PIL.ImageDraw.Draw(frame)
    draw.text(pos, text, fill=color, font=PIL.ImageFont.truetype(font, fontsize))
    return frame

def display_video_with_rewards(frames, rewards, framerate=30):
    assert len(frames) == len(rewards), f"Size of frames and rewards does not match ({len(frames)}, {len(rewards)})!"
    
    for frame_id, reward in enumerate(rewards):
        frames[frame_id] = np.array(add_text_to_frame(frames[frame_id],
                        f"#{frame_id}",
                        pos=(5, 5), color=(0, 0, 0), fontsize=18))
        frames[frame_id] = np.array(add_text_to_frame(frames[frame_id],
                        f"Reward {reward:.2f}",
                        pos=(15, 25), color=(99, 207, 163), fontsize=24))

    anim = _create_animation(frames, framerate=framerate)
    return HTML(anim.to_html5_video())


def pointing_evaluation(env, policy, target_position, target_radius, n_trials=1, max_episode_steps=np.inf):
    # Create empty list which all frames of the forward simulation will be appended to
    frame_collection = []

    # Create empty list which all rewards of the forward simulation will be appended to
    reward_collection = []

    # Create empty list which all success flags of the forward simulation will be appended to
    success_collection = []

    for trial in range(n_trials):

        # Reset with some seed for debugging purposes
        obs, info = env.reset()
        reward_collection += [np.nan]  #no reward available at initial state

        # Manually update target position
        env.task._spawn_target(env._model, env._data, new_position=target_position, new_radius=target_radius)

        terminated = False
        truncated = False
        success = False
        step = 0

        while not (terminated or truncated):  #for step in range(num_steps):
            step += 1

            # # choose random action from action space
            # action = env.action_space.sample()
            
            # get actions from policy
            action, _internal_policy_state = policy.predict(obs, deterministic=True)
            
            # apply the action
            obs, reward, terminated, truncated, info = env.step(action)
            success = success or info["target_hit"]

            if step >= max_episode_steps:
                truncated = True  #truncate from outer loop
            
            # store received reward
            reward_collection.append(reward)
            
            # If the epsiode is up, then stop
            if terminated or truncated:
                frame_collection.extend(env.render()) #with render_mode="rgb_array_list", env.render() returns a list of all frames since last call of reset()
                break

            # if step >= max_episode_steps/2:
            #     env.perception.perception_modules[0]._camera_active = False
            # if step >= 3*max_episode_steps/4:
            #     env.perception.perception_modules[0]._camera_active = True

        # Store whether trial was successful
        success_collection.append(success)

        # Also store remaining frames of unfinished last episode
        frame_collection.extend(env.render())

    # Calculate success rate over n_trials
    success_rate = sum(success_collection)/len(success_collection)

    # Close the env
    env.close()

    return success_rate