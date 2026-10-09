"""Real MuJoCo asset contracts; these checks never connect to robot hardware."""

import hashlib
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "microduck/src/model/mjcf"
TRAIN = REPO / "mjlab_microduck/src/mjlab_microduck/robot/microduck"
sys.path.insert(0, str(REPO / "microduck/src"))

from constants import MOTOR_TO_ID, OBSERVATION_DOF_ORDER
from imu_reader import imu_quat_to_body, quat_apply_inverse


class ModelAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = mujoco.MjModel.from_xml_path(str(DEPLOY / "scene.xml"))
        cls.training = mujoco.MjModel.from_xml_path(str(TRAIN / "robot_walk.xml"))

    def test_real_models_match_motor_and_policy_joint_contract(self):
        for model in (self.deployment, self.training):
            with self.subTest(model=model):
                self.assertEqual((model.nu, model.nq, model.nv, model.njnt), (14, 21, 20, 15))
                actuators = [model.actuator(i).name for i in range(model.nu)]
                self.assertEqual(set(actuators), set(MOTOR_TO_ID))
                self.assertEqual(actuators, OBSERVATION_DOF_ORDER)
                for name in MOTOR_TO_ID:
                    joint = model.joint(name)
                    self.assertEqual(model.jnt_type[joint.id], mujoco.mjtJoint.mjJNT_HINGE)
                    actuator = model.actuator(name)
                    self.assertEqual(model.actuator_trnid[actuator.id, 0], joint.id)

    def test_real_geometry_and_sensor_contract(self):
        for model in (self.deployment, self.training):
            with self.subTest(model=model):
                self.assertEqual(model.nmesh, 38)
                self.assertTrue(np.all(model.mesh_vertnum > 0))
                self.assertTrue(np.all(model.mesh_facenum > 0))
                self.assertGreater(float(model.body_mass.sum()), 0.5)
                for name in ("left_foot_collision", "right_foot_collision"):
                    self.assertEqual(model.geom_type[model.geom(name).id], mujoco.mjtGeom.mjGEOM_MESH)
                for name, kind, dim in (
                    ("orientation", mujoco.mjtSensor.mjSENS_FRAMEQUAT, 4),
                    ("angular-velocity", mujoco.mjtSensor.mjSENS_GYRO, 3),
                    ("imu_accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER, 3),
                ):
                    sensor = model.sensor(name)
                    self.assertEqual(model.sensor_type[sensor.id], kind)
                    self.assertEqual(model.sensor_dim[sensor.id], dim)
                    self.assertEqual(model.sensor_objid[sensor.id], model.site("imu").id)

    def test_deployment_zero_pose_policy_up_and_body_down(self):
        data = mujoco.MjData(self.deployment)
        mujoco.mj_forward(self.deployment, data)
        sensor_quat = data.sensor("orientation").data.copy()
        body_quat = imu_quat_to_body(tuple(sensor_quat))
        np.testing.assert_allclose(body_quat, [1, 0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(
            quat_apply_inverse(sensor_quat, [0, 0, 1]), [1, 0, 0], atol=1e-7
        )
        np.testing.assert_allclose(
            quat_apply_inverse(body_quat, [0, 0, -1]), [0, 0, -1], atol=1e-7
        )

    def test_training_keeps_official_body_marker(self):
        data = mujoco.MjData(self.training)
        mujoco.mj_forward(self.training, data)
        np.testing.assert_allclose(data.sensor("orientation").data, [1, 0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(
            self.training.site_quat[self.training.site("imu").id], [1, 0, 0, 0], atol=1e-7
        )

    def test_mount_maps_tilted_sensor_orientation_back_to_body(self):
        data = mujoco.MjData(self.deployment)
        body_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(body_quat, np.array([1., 0., 0.]), 0.3)
        data.qpos[3:7] = body_quat
        mujoco.mj_forward(self.deployment, data)
        np.testing.assert_allclose(
            imu_quat_to_body(tuple(data.sensor("orientation").data)), body_quat, atol=1e-7
        )

    def test_manifest_proves_mesh_bytes_and_all_local_asset_hashes(self):
        ledger = json.loads((DEPLOY / "microduck_14dof/source-manifest.json").read_text())
        self.assertEqual(ledger["source_commit"], "1e79c29c97d8b38aee9eefde77a545860ba7658e")
        for entry in ledger["files"]:
            path = REPO / entry["local_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["local_sha256"], path)
            if path.suffix == ".stl":
                self.assertEqual(entry["source_sha256"], entry["local_sha256"], path)
        source_meshes = {e.attrib["file"] for e in ET.parse(TRAIN / "robot_walk.xml").findall("./asset/mesh")}
        self.assertEqual(len(source_meshes), 38)
        self.assertEqual(source_meshes, {p.name for p in (TRAIN / "assets").glob("*.stl")})

    def test_legacy_19_actuator_model_is_preserved(self):
        legacy = ET.parse(DEPLOY / "robot.xml")
        self.assertEqual(len(legacy.findall("./actuator/*")), 19)
        self.assertNotEqual({e.attrib["name"] for e in legacy.findall("./actuator/*")}, set(MOTOR_TO_ID))
        ledger = json.loads((DEPLOY / "microduck_14dof/source-manifest.json").read_text())
        self.assertEqual(
            hashlib.sha256((DEPLOY / "robot.xml").read_bytes()).hexdigest(),
            ledger["legacy_model_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
