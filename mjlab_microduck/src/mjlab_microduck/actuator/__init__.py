# SPDX-License-Identifier: Apache-2.0
"""Walk-only BAM configuration with an explicit voltage-sag API adapter."""

from dataclasses import fields

from bam.model import load_model
from mjlab.managers.event_manager import requires_model_fields

from mjlab_microduck.actuator.friction_dr_bam import (
    FrictionDRBamActuator,
    FrictionDRBamActuatorCfg,
)


@requires_model_fields("dof_frictionloss", "dof_damping")
def bam_init(env, env_ids=None) -> None:
    """Request BAM's per-world fields without depending on a BAM release hook.

    Git BAM 1.0.0 computes friction in torque space and has no ``bam_init``;
    PyPI BAM 1.0.2 writes native solver fields and needs both fields expanded.
    The same MjLab declaration works for either implementation. The event body
    intentionally has no mutation; MjLab reads the decorator before setup.
    """
    del env, env_ids


def make_walk_actuator_cfg() -> FrictionDRBamActuatorCfg:
    """Preserve upstream V_drop = gain * sum(abs(torque)) across BAM APIs.

    BAM 1.0.2 renamed the setting to battery/wire resistance and divides summed
    torque by Kt to estimate current. R = gain * Kt preserves the old voltage
    drop. This adapts a parameter's units, not the robot's firmware P gain.
    """
    kwargs = dict(
        motor_name="xl330",
        model="m6",
        target_names_expr=(r"^(?!passive_).*",),
        kp_fw=200.0,
        vin_range=(6.5, 8.2),
        vin_min=6.0,
        delay_min_lag=3,
        delay_max_lag=6,
    )
    gain_range = (0.0, 0.2)
    field_names = {field.name for field in fields(FrictionDRBamActuatorCfg)}
    if "vin_drop_gain_range" in field_names:
        kwargs["vin_drop_gain_range"] = gain_range
    elif "vin_drop_resistance_range" in field_names:
        motor_model = load_model(motor_name="xl330", model="m6")
        kwargs["vin_drop_resistance_range"] = tuple(
            gain * motor_model.kt.value for gain in gain_range
        )
    else:
        raise RuntimeError("BAM does not expose a supported voltage-sag parameter")
    return FrictionDRBamActuatorCfg(**kwargs)


__all__ = [
    "FrictionDRBamActuator",
    "FrictionDRBamActuatorCfg",
    "make_walk_actuator_cfg",
    "bam_init",
]
