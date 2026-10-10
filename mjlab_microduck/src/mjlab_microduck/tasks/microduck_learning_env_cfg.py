# SPDX-License-Identifier: Apache-2.0
"""A single deterministic standing task for learning the PPO pipeline on CPU.

This configuration deliberately sacrifices gait quality for a small, inspectable
experiment. It is not a hardware policy or a reduced professional training run.
"""

from copy import deepcopy

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

TASK = "Mjlab-Learning-Stand-Microduck"


class StandingCommand(UniformVelocityCommand):
    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        # Upstream's joystick assumes nonzero ranges. Keep this teaching task
        # fixed at zero, including in the viewer, instead of widening the task.
        with server.gui.add_folder(name.capitalize()):
            server.gui.add_markdown("固定站立任务：目标速度为零。学习模型仅用于仿真。")


class StandingCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return StandingCommand(self, env)


def make_microduck_learning_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=False)
    cfg.scene.num_envs = 1
    cfg.scene.sensors = ()
    cfg.episode_length_s = 2.0
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.rewards = {
        name: cfg.rewards[name]
        for name in ("upright", "track_linear_velocity", "action_rate_l2")
    }
    # Both networks see the same 51 values: no privileged critic or history.
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    for term in actor.terms.values():
        term.noise = None
        term.delay_min_lag = term.delay_max_lag = 0
    cfg.observations["critic"] = deepcopy(actor)
    cfg.commands["twist"] = StandingCommandCfg(
        entity_name="robot",
        resampling_time_range=(2.0, 2.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0), heading=None,
        ),
    )
    cfg.events = {
        name: cfg.events[name]
        for name in ("bam_init", "reset_base", "reset_robot_joints")
    }
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.125, 0.125),
        "yaw": (0.0, 0.0),
    }
    # Reuse the robot and BAM physics, with fixed voltage and no random delays.
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.vin_range = (7.4, 7.4)
        actuator.delay_min_lag = actuator.delay_max_lag = 0
        for field in ("vin_drop_gain_range", "vin_drop_resistance_range"):
            if hasattr(actuator, field):
                setattr(actuator, field, (0.0, 0.0))
    return cfg


MicroduckLearningRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(64, 32), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1,
                          "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 32), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
        num_learning_epochs=2, num_mini_batches=1, learning_rate=1.0e-3,
        schedule="fixed", entropy_coef=0.0,
    ),
    experiment_name="microduck_learning_stand",
    logger="tensorboard", upload_model=False, save_interval=1,
    num_steps_per_env=24, max_iterations=3,
)
