# MyoInteract Studio as a Docker Space (see README.md). The GPU driver comes from the Space's hardware.
FROM python:3.12-slim

# git: install from GitHub; EGL/GL: MuJoCo's offscreen renderer (task previews); ffmpeg: rollout videos.
RUN apt-get update && apt-get install -y --no-install-recommends git libegl1 libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000.
RUN useradd -m -u 1000 user && mkdir -p /data && chown user /data
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH MUJOCO_GL=egl MYOINTERACT_LOGGER=tensorboard
WORKDIR /home/user

RUN pip install --no-cache-dir --user \
      "myoInteract[all] @ git+https://github.com/MyoInteract/MyoInteract.git@v1.0.0"

EXPOSE 7860
# /data is the Space's persistent storage, if it has one.
CMD ["sh", "-c", "myo_studio serve --host 0.0.0.0 --port 7860 --no-browser --behind-proxy --root $( [ -w /data ] && echo /data || echo $HOME/myoInteract_data )"]
