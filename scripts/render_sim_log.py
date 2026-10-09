#!/usr/bin/env python3
"""Render recorded MuJoCo states, without running policy or connecting to hardware."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg is required to encode the preview")
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    header = rows[0]
    if header.get("evidence") != "LOCAL_SIM_POLICY_CLOSED_LOOP" or header.get("hardware_verified") is not False:
        parser.error("Expected a local simulation log")
    ticks = [row for row in rows if row.get("type") == "tick"]
    delays = header.get("delays", {})
    delay_label = "no sensor delay" if delays and not any(delays.values()) else f"delays={delays}"
    model = mujoco.MjModel.from_xml_path(str(ROOT / "microduck/src/model/mjcf/scene.xml"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .60, 135, -18
    stride = max(1, round(header["hz"] / 10))
    fps = header["hz"] / stride
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output)], stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for tick in ticks[::stride]:
                positions = np.asarray(tick["qpos_after_step"])
                if positions.shape != (model.nq,) or not np.isfinite(positions).all():
                    raise ValueError("Invalid recorded joint positions")
                data.qpos[:] = positions
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [data.qpos[0], data.qpos[1], .14]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 43), fill="black")
                draw.text((10, 5), "LOCAL CPU SIMULATION - NOT HARDWARE", fill="white")
                draw.text((10, 23), f't={tick["simulation_time_s"]:.2f}s | {delay_label} | KP={header.get("policy_firmware_kp", "see log")}', fill="white")
                process.stdin.write(frame.tobytes())
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass  # An encoder failure is reported by its exit code below.
        finally:
            result = process.wait()
    if result:
        raise RuntimeError(f"Video encoding failed: {result}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
