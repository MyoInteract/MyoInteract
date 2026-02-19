import os

def get_latest_run_path(base_path="logs/"):
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and os.path.isdir(os.path.join(base_path, d, "checkpoints")) and len(os.listdir(os.path.join(base_path, d, "checkpoints"))) > 1]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_path, d)))
    return os.path.join(base_path, latest_subdir, "checkpoints")
