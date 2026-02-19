import torch

def check_mjx_usable():
    USE_MJX = torch.cuda.is_available()
    if USE_MJX:
        try:
            import jax
            import mujoco
            from mujoco import mjx
        except ImportError as e:
            print(f"Some module required for MuJoCo-MJX could not be found: {e}")
            USE_MJX = False
    return USE_MJX

def mujoco_to_mjx(model, data):
    mjx_usable = check_mjx_usable()
    if mjx_usable:
        from mujoco import mjx
        mjx_model = mjx.put_model(model)
        mjx_data = mjx.put_data(model, data)
    else:
        ## return non-MJX model and data object and continue without MJX
        mjx_model = model
        mjx_data = data
    return mjx_model, mjx_data
