# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckVelocityRlCfg,
)
from mjlab_microduck.tasks.microduck_learning_env_cfg import (
    TASK as LEARNING_TASK,
    MicroduckLearningRlCfg,
    make_microduck_learning_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Microduck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckVelocityRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# An independent CPU teaching task; never changes the velocity task or assets.
register_mjlab_task(
    task_id=LEARNING_TASK,
    env_cfg=make_microduck_learning_env_cfg(),
    play_env_cfg=make_microduck_learning_env_cfg(play=True),
    rl_cfg=MicroduckLearningRlCfg,
    runner_cls=MjlabOnPolicyRunner,
)
