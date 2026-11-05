# Adapted from : https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/python/torchcontrol/planning/min_jerk.py

import numpy as np

from myosuite.utils.movement_metrics import minimum_jerk

def _min_jerk_spaces(N: int, T: float):
    """
    Generates a 1-dim minimum jerk trajectory from 0 to 1 in N steps & T seconds.
    Assumes zero velocity & acceleration at start & goal.
    The resulting trajectories can be scaled for different start & goals.
    Args:
        N: Length of resulting trajectory in steps
        T: Duration of resulting trajectory in seconds
    Returns:
        p_traj: Position trajectory of shape (N,)
        pd_traj: Velocity trajectory of shape (N,)
        pdd_traj: Acceleration trajectory of shape (N,)
    """
    assert N > 1, "Number of planning steps must be larger than 1."

    # Same polynomial as the movement-metrics reference (Flash & Hogan 1985).
    traj = minimum_jerk(0.0, 1.0, T, np.linspace(0, T, N))

    return traj.position, traj.velocity, traj.acceleration

def generate_joint_space_min_jerk(
    start, goal, time_to_go: float, dt: float):
    """
    Primitive joint space minimum jerk trajectory planner.
    Assumes zero velocity & acceleration at start & goal.
    Args:
        start: Start joint position of shape (N,)
        goal: Goal joint position of shape (N,)
        time_to_go: Trajectory duration in seconds
        hz: Frequency of output trajectory
    Returns:
        waypoints: List of waypoints
    """
    steps =  int(time_to_go/dt)

    p_traj, pd_traj, pdd_traj = _min_jerk_spaces(steps, time_to_go)

    D = goal - start
    q_traj = start[None, :] + D[None, :] * p_traj[:, None]
    qd_traj = D[None, :] * pd_traj[:, None]
    qdd_traj = D[None, :] * pdd_traj[:, None]

    waypoints = [
        {
            "time_from_start": i * dt,
            "position": q_traj[i, :],
            "velocity": qd_traj[i, :],
            "acceleration": qdd_traj[i, :],
        }
        for i in range(steps)
    ]

    return waypoints