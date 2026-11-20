import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg

from myoInteract.active_vision import concat_dual_acuity_streams
from myoInteract.common import MyoComponent

from ..vision_registry import myo_register_vision
from .dual_acuity_config import DualAcuityVisionConfig


def _resolve_env_fixation_uv(
    env: ManagerBasedRlEnv, default_uv: tuple[float, float]
) -> object:
    gaze_state = getattr(env, "_active_gaze_state", None)
    if gaze_state is not None:
        return gaze_state.uv
    gaze_uv = getattr(env, "_active_gaze_uv", None)
    if gaze_uv is not None:
        return gaze_uv
    return default_uv


def dual_acuity_rgb(
    env: ManagerBasedRlEnv,
    periphery_sensor_name: str,
    fovea_sensor_name: str | None,
    stream_height: int,
    stream_width: int,
    fov_ratio: float,
    default_fixation_uv: tuple[float, float],
) -> torch.Tensor:
    periphery_sensor: CameraSensor = env.scene[periphery_sensor_name]
    raw_wide = periphery_sensor.data.rgb
    assert raw_wide is not None, f"Camera '{periphery_sensor_name}' has no RGB data"
    wide = (
        raw_wide.permute(0, 3, 1, 2)
        .to(dtype=torch.float32, device=env.device)
        .div_(255.0)
    )

    narrow = None
    if fovea_sensor_name is not None:
        fovea_sensor: CameraSensor = env.scene[fovea_sensor_name]
        raw_narrow = fovea_sensor.data.rgb
        assert raw_narrow is not None, f"Camera '{fovea_sensor_name}' has no RGB data"
        narrow = (
            raw_narrow.permute(0, 3, 1, 2)
            .to(dtype=torch.float32, device=env.device)
            .div_(255.0)
        )

    uv = _resolve_env_fixation_uv(env, default_fixation_uv)
    return concat_dual_acuity_streams(
        wide,
        uv,
        stream_height=stream_height,
        stream_width=stream_width,
        fov_ratio=fov_ratio,
        narrow_images=narrow,
    )


def dual_acuity_depth(
    env: ManagerBasedRlEnv,
    periphery_sensor_name: str,
    fovea_sensor_name: str | None,
    stream_height: int,
    stream_width: int,
    fov_ratio: float,
    cutoff_distance: float,
    min_depth: float,
    default_fixation_uv: tuple[float, float],
) -> torch.Tensor:
    def _norm_depth(sensor_name: str) -> torch.Tensor:
        sensor: CameraSensor = env.scene[sensor_name]
        raw = sensor.data.depth
        assert raw is not None, f"Camera '{sensor_name}' has no depth data"
        return (
            raw.permute(0, 3, 1, 2)
            .to(dtype=torch.float32, device=env.device)
            .clamp_(min_depth, cutoff_distance)
            .sub_(min_depth)
            .div_(cutoff_distance - min_depth)
        )

    wide = _norm_depth(periphery_sensor_name)
    narrow = _norm_depth(fovea_sensor_name) if fovea_sensor_name is not None else None
    uv = _resolve_env_fixation_uv(env, default_fixation_uv)
    return concat_dual_acuity_streams(
        wide,
        uv,
        stream_height=stream_height,
        stream_width=stream_width,
        fov_ratio=fov_ratio,
        narrow_images=narrow,
    )


@myo_register_vision("dual_acuity")
class DualAcuityVisionComponent(MyoComponent):
    def __init__(self, cfg: DualAcuityVisionConfig):
        self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        del play
        data_types: list[str] = []
        terms: dict[str, ObservationTermCfg] = {}
        fovea_sensor = (
            "dual_acuity_fovea_sensor"
            if self.cfg.fovea_camera_name is not None
            else None
        )

        if self.cfg.use_rgb:
            data_types.append("rgb")
            terms["dual_rgb"] = ObservationTermCfg(
                func=dual_acuity_rgb,
                params={
                    "periphery_sensor_name": "dual_acuity_periphery_sensor",
                    "fovea_sensor_name": fovea_sensor,
                    "stream_height": self.cfg.stream_height,
                    "stream_width": self.cfg.stream_width,
                    "fov_ratio": self.cfg.fov_ratio,
                    "default_fixation_uv": self.cfg.default_fixation_uv,
                },
            )

        if self.cfg.use_depth:
            data_types.append("depth")
            terms["dual_depth"] = ObservationTermCfg(
                func=dual_acuity_depth,
                params={
                    "periphery_sensor_name": "dual_acuity_periphery_sensor",
                    "fovea_sensor_name": fovea_sensor,
                    "stream_height": self.cfg.stream_height,
                    "stream_width": self.cfg.stream_width,
                    "fov_ratio": self.cfg.fov_ratio,
                    "cutoff_distance": self.cfg.depth_cutoff,
                    "min_depth": self.cfg.depth_min,
                    "default_fixation_uv": self.cfg.default_fixation_uv,
                },
            )

        sensors = [
            CameraSensorCfg(
                name="dual_acuity_periphery_sensor",
                camera_name=self.cfg.periphery_camera_name,
                width=self.cfg.width,
                height=self.cfg.height,
                data_types=tuple(data_types),
                use_textures=self.cfg.use_textures,
                use_shadows=self.cfg.use_shadows,
                enabled_geom_groups=self.cfg.enabled_geom_groups,
            )
        ]
        if self.cfg.fovea_camera_name is not None:
            sensors.append(
                CameraSensorCfg(
                    name="dual_acuity_fovea_sensor",
                    camera_name=self.cfg.fovea_camera_name,
                    width=self.cfg.stream_width,
                    height=self.cfg.stream_height,
                    data_types=tuple(data_types),
                    use_textures=self.cfg.use_textures,
                    use_shadows=self.cfg.use_shadows,
                    enabled_geom_groups=self.cfg.enabled_geom_groups,
                )
            )
        cfg.scene.sensors += tuple(sensors)
        cfg.observations["vision_cnn"] = ObservationGroupCfg(
            terms=terms, concatenate_dim=0
        )

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        if self.cfg.spatial_softmax:
            cfg.actor.class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
            cfg.actor.cnn_cfg = {
                "output_channels": [16, 64, 64, 128],
                "kernel_size": [5, 3, 3, 3],
                "stride": [2, 2, 1, 1],
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
                "kernel_size": [5, 3, 3, 3],
                "stride": [2, 2, 1, 1],
                "dilation": [1, 1, 1, 1],
                "padding": "none",
                "norm": ["layer", "layer", "layer", "layer"],
                "activation": "elu",
                "max_pool": [False, False, False, False],
                "global_pool": "avg",
                "flatten": True,
            }
        cfg.obs_groups = {
            "actor": ("agent_state", "vision_cnn"),
            "critic": ("agent_state", "task_state"),
        }
