# Restored official walk assets

The robot XML, scene, export configuration and 38 referenced STL meshes under
`microduck/` are unmodified copies from `pollen-robotics/microduck_rl` at commit
`1e79c29c97d8b38aee9eefde77a545860ba7658e`. Only the walk subset of robot and
actuator Python code is restored. `MICRODUCK_ROBOT_CFG` aliases the upstream
`MICRODUCK_WALK_ROBOT_CFG`; the existing task's reward design is preserved.

The complete provenance, per-file hash ledger, deployment IMU adaptation,
BAM voltage-sag API conversion and verification limits are in
[`MODEL_SOURCE.md`](../../../../microduck/src/model/mjcf/microduck_14dof/MODEL_SOURCE.md).
This source snapshot is not proven to be the deployed 20260904 ONNX's exact
training assets.

Upstream software is Apache 2.0 (`UPSTREAM_LICENSE.txt`). Upstream README assigns
hardware designs to Creative Commons BY-SA-NC without specifying a version;
`UPSTREAM_LICENSE_NOTICE.txt` preserves that notice. The meshes are hardware
design assets, and the Apache software license does not replace their license.

## Training BAM release compatibility

The declared Git source resolves to BAM 1.0.0 at
`0007411bd82c48f1bc12ae8382f98e4d8b879c95` on branch `doc`, while deployment
uses PyPI BAM 1.0.2. BAM 1.0.0 exposes `vin_drop_gain_range` and computes friction
as torque; it has no `bam.mjlab.bam_init`. BAM 1.0.2 exposes
`vin_drop_resistance_range` and writes native solver friction/damping fields.
The local actuator factory supports both parameter APIs. The local decorated
`bam_init` requests MjLab per-world `dof_frictionloss` and `dof_damping`
allocation directly, without importing a release-specific BAM hook. Its body
is intentionally a no-op. The extra field allocation is harmless for the
torque-space implementation and required for the native-field implementation.

Actual configuration/Entity compilation is checked with the installed declared
training environment using `tests/test_robot_contract.py`. Import MjLab before
robot submodules in ad hoc checks so MjLab's entry-point discovery does not
encounter a partially initialized robot configuration.

On 2026-10-09, the installed MjLab 1.3.0 / Git BAM 1.0.0 training environment
passed all three training contracts. The unmodified-capacity `play=True`
configuration constructed on CPU with one environment, reset, and completed
three zero-action steps: actor `(1,51)`, critic `(1,66)`, 14 actions, finite
observations and rewards, and no termination or truncation. The control period
was 0.02 s (four 0.005 s physics substeps). This establishes CPU environment
execution, not policy learning, long-horizon stability, original-ONNX lineage,
CUDA training or hardware execution. PyTorch reported CUDA unavailable on this
Mac. MjLab emits a warning that the upstream joint-target regex also matches
site names; its explicitly joint-typed actuator still compiles exactly 14
actuators. The source regex is preserved.
