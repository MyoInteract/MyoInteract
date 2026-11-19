import torch
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.managers import ObservationTermCfg, ObservationGroupCfg
from mjlab.sensor import CameraSensorCfg, CameraSensor

from myoInteract.common import MyoComponent
from myoInteract.foveation import FoveationParams, foveate_tensor
from .cnn_vision_config import CnnVisionConfig
from ..vision_registry import myo_register_vision


def cnn_camera_rgb(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foveation: FoveationParams | None = None,
) -> torch.Tensor:
    """Modified version of mjlab.tasks.manipulation::camera_rgb with optional foveation."""
    sensor: CameraSensor = env.scene[sensor_name]

    raw_rgb = sensor.data.rgb  # (B, H, W, 3)
    assert raw_rgb is not None, f"Camera '{sensor_name}' has no RGB data"

    rgb_data = raw_rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)
    rgb_data = rgb_data.to(dtype=torch.float32, device=env.device)
    rgb_data = rgb_data.div_(255.0)

    if foveation is not None:
        rgb_data = foveate_tensor(rgb_data, foveation)

    return rgb_data


def cnn_camera_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float,
    min_depth: float = 0.01,
    foveation: FoveationParams | None = None,
) -> torch.Tensor:
    """Modified version of mjlab.tasks.manipulation::camera_depth with optional foveation."""
    sensor: CameraSensor = env.scene[sensor_name]

    depth_raw = sensor.data.depth  # (B, H, W, 1)
    assert depth_raw is not None, f"Camera '{sensor_name}' has no depth data"

    depth_data = (
        depth_raw.permute(0, 3, 1, 2)
        .clamp_(min_depth, cutoff_distance)
        .sub_(min_depth)
        .div_(cutoff_distance - min_depth)
    )

    if foveation is not None:
        depth_data = foveate_tensor(depth_data, foveation)

    return depth_data


@myo_register_vision("cnn")
class CnnVisionComponent(MyoComponent):
    def __init__(self, cfg: CnnVisionConfig):
        self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        data_types: list[str] = []
        terms: dict[str, ObservationTermCfg] = {}
        foveation = self.cfg.foveation_params()

        if self.cfg.use_rgb:
            data_types.append("rgb")
            rgb_params: dict[str, object] = {"sensor_name": "cnn_sensor"}
            if foveation is not None:
                rgb_params["foveation"] = foveation
            terms["cnn_rgb"] = ObservationTermCfg(
                func=cnn_camera_rgb, params=rgb_params
            )

        if self.cfg.use_depth:
            data_types.append("depth")
            depth_params: dict[str, object] = {
                "sensor_name": "cnn_sensor",
                "cutoff_distance": self.cfg.depth_cutoff,
                "min_depth": self.cfg.depth_min,
            }
            if foveation is not None:
                depth_params["foveation"] = foveation
            terms["cnn_depth"] = ObservationTermCfg(
                func=cnn_camera_depth,
                params=depth_params,
            )

        cfg.scene.sensors += (
            CameraSensorCfg(
                name="cnn_sensor",
                camera_name=self.cfg.camera_name,
                width=self.cfg.width,
                height=self.cfg.height,
                data_types=tuple(data_types),
                use_textures=self.cfg.use_textures,
                use_shadows=self.cfg.use_shadows,
                enabled_geom_groups=self.cfg.enabled_geom_groups,
            ),
        )

        cfg.observations["vision_cnn"] = ObservationGroupCfg(
            terms=terms, concatenate_dim=0
        )

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        if self.cfg.spatial_softmax:
            cfg.actor.class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"

            cfg.actor.cnn_cfg = {
                "output_channels": [16, 64, 64, 128],
                "kernel_size": [8, 4, 3, 3],
                "stride": [4, 2, 1, 1],
                "dilation": [1, 1, 1, 1],
                "padding": "none",
                "norm": ["layer", "layer", "layer", "layer"],
                "activation": "elu",
                "max_pool": [False, False, False, False],
                "global_pool": "avg",
                "flatten": True,
                "spatial_softmax": True,
                "spatial_softmax_temperature": 0.2,
            }
        else:
            cfg.actor.class_name = "CNNModel"

            cfg.actor.cnn_cfg = {
                "output_channels": [16, 64, 64, 128],
                "kernel_size": [8, 4, 3, 3],
                "stride": [4, 2, 1, 1],
                "dilation": [1, 1, 1, 1],
                "padding": "none",
                "norm": ["layer", "layer", "layer", "layer"],
                "activation": "elu",
                "max_pool": [False, False, False, False],
                # default non-spatial readout
                "global_pool": "avg",
                "flatten": True,
            }

        cfg.obs_groups = {
            "actor": ("agent_state", "vision_cnn"),
            "critic": ("agent_state", "task_state"),
        }
