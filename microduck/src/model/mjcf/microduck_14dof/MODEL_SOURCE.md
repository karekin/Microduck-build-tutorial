# 14DOF Microduck model provenance

This restores the official **walk** model from
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl/tree/1e79c29c97d8b38aee9eefde77a545860ba7658e),
commit `1e79c29c97d8b38aee9eefde77a545860ba7658e` (archive dated 2026-08-25).
The source archive was already available locally as `microduck_rl-official.zip`.
`source-manifest.json` records the archive hash and each imported source/local file
hash. The generated MJCF preserves its original Onshape source comment.

## Scope and changes

- All 38 STL meshes referenced by `robot_walk.xml` are copied byte for byte.
  Masses, inertias, joint axes, ranges, collision shapes, actuator names and order
  retain the official model. The result has 14 hinge joints plus one floating
  root: `nu=14`, `nq=21`, `nv=20`.
- The tutorial's existing `robot.xml` and legacy `assets/` remain intact. That
  file is a different 19-actuator model and is no longer the default scene model.
- Deployment `robot_walk.xml` changes only the trunk `imu` site's quaternion to
  `(w,x,y,z)=(0.7071068,0,-0.7071068,0)`, a -90 degree rotation about Y. It is the
  inverse of deployment `IMU_MOUNT_QUAT=(0.7071068,0,0.7071068,0)`, because the
  observer reconstructs `body_quat = sensor_quat * IMU_MOUNT_QUAT`. At the upright
  zero pose, policy projected **up** is `(1,0,0)` and body projected **down** is
  `(0,0,-1)`. These two vector conventions must not be interchanged.
- The enclosing `../scene.xml` includes this 14DOF model and selects
  `meshdir="microduck_14dof/assets"` with `strippath="true"`. This resolves meshes
  from the nested include for both relative and absolute entry-point paths.
- Training assets in `mjlab_microduck/src/mjlab_microduck/robot/microduck/` keep
  the official XML byte for byte, including its body-aligned IMU site. The
  tutorial's existing training observation functions rotate body observations
  at the observation layer; rotating the training site again is unnecessary.
- Training robot constants retain only the upstream walk configuration, HOME
  pose and collision configuration. `MICRODUCK_ROBOT_CFG` is an alias of
  `MICRODUCK_WALK_ROBOT_CFG` for the existing tutorial task. No additional task,
  roller, ball or backlash model is imported.
- The restored actuator module contains the upstream friction-randomization
  subclass needed by the walk configuration. Its factory preserves upstream
  firmware `kp_fw=200`, voltage range `(6.5,8.2)`, delay `(3,6)` and voltage floor
  `6.0`. BAM 1.0.2's current API expresses sag as `R * sum(abs(torque)) / Kt`;
  the adapter maps the upstream gain range `(0,0.2)` to `R = gain * Kt` using the
  packaged XL330 m6 model. Old BAM's `vin_drop_gain_range` API is also recognized.
  Unsupported sag APIs fail explicitly.
- The existing velocity task registers BAM's `bam_init` startup event. Its
  `requires_model_fields("dof_frictionloss", "dof_damping")` annotation requests
  per-environment field expansion before actuator stepping. No reward,
  termination or curriculum definition changes. The pre-existing XML
  `dof_friction_randomization` event is preserved; BAM owns the friction budget,
  so that event alone does not randomize BAM's friction-scale multiplier.

## Verification boundary

MuJoCo asset contract tests compile both real models, check all motor/actuator
names and policy ordering, real mesh and sensor data, deployment zero/tilted
orientation, training's unmodified marker, and imported file hashes.

The MjLab 1.3.0 wheel source was inspected for `EntityCfg`,
`EntityArticulationInfoCfg`, `CollisionCfg`, actuator `build`/`initialize` and the
tutorial's imported symbols. This is source/API verification. It does not prove
a GPU training run or even a complete training-runtime import. BAM startup field
expansion and training events require a separate environment-level check with
the declared training dependencies installed.

The supplied deployment ONNX identifies training date `20260904`; the official
model archive predates it. No manifest binds that ONNX to this exact XML, mesh,
actuator configuration or reward code. The restored source therefore must **not**
be described as the ONNX's exact original training assets, successful sim-to-real
validation, or permission to run hardware. Deployment `KP_RL=125` also differs
from this official walk source's `kp_fw=200`; this restoration does not change
either value or claim their equivalence.

## Attribution and licenses

The upstream README's License section states that software is Apache 2.0 and
hardware design files are Creative Commons **BY-SA-NC**. The README does not
specify a Creative Commons version; this restoration does not infer one.
`UPSTREAM_LICENSE.txt` retains the upstream Apache 2.0 text, and
`UPSTREAM_LICENSE_NOTICE.txt` retains that README license section and source URL.
Copied 3D meshes remain subject to the hardware-design license, including its
attribution, share-alike and noncommercial conditions. Copying the Apache text
does not relicense meshes as Apache. Existing tutorial deployment source retains
its own GPL notices.
