"""Real production speed-keyboard + heading-assist callback physics replays.

This script creates LOCAL_SIM evidence; it never changes production code,
physical state beyond the normal controller, motor parameters, or policy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imu_reader import quat_apply_inverse
from moves.walk import WalkMove
from scheduler import Scheduler
from sim.low_speed_input import LowSpeedKeyboardInput
from sim.mujoco_controller import MuJoCoController
from test_keyboard_closed_loop import EventKeyboardInput, UP, RIGHT, X, R

CASES = {
    'A': (.10, 1500, [(0, UP, 2)]),
    'B': (.05, 3500, [(0, UP, 1), (500, ord('9'), 1), (1250, RIGHT, 2),
                     (1500, X, 1), (2000, UP, 2), (2750, ord('8'), 1), (3000, R, 1)]),
    'C': (.10, 1750, [(0, UP, 2), (750, RIGHT, 2), (1000, X, 1)]),
}


def wrapped(value):
    return math.atan2(math.sin(value), math.cos(value))


def run_scenario(name, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'production_speed_keyboard_{name}.jsonl'
    if path.exists():
        raise FileExistsError(f'Preserve existing evidence; select a fresh --output-dir: {path}')
    forward_limit, tick_limit, events = CASES[name]
    stop_path = folder / f'production_speed_keyboard_{name}.stop'
    controller = MuJoCoController(str(SRC/'model/mjcf/scene.xml'), headless=True,
                                  scheduler_driven=True, frequency_hz=50., policy_kp=200,
                                  initial_height_m=.12, physics_timestep_s=.005,
                                  delay_act_steps=0, delay_pos_ticks=0, delay_vel_ticks=0,
                                  delay_gyro_ticks=0, delay_quat_ticks=0,
                                  stop_flag_path=str(stop_path))
    keyboard = LowSpeedKeyboardInput(time_source=lambda: controller.simulation_time,
                                    stop_flag_path=str(stop_path), initial_moves={'walk'},
                                    forward_limit=forward_limit, heading_source=controller.read_body_heading)
    source = EventKeyboardInput(keyboard, events)
    controller._reset_source = source
    walk = WalkMove(controller=controller)
    scheduler = Scheduler(controller=controller, input_source=source, moves={'walk': walk},
                          stop_flag_path=str(stop_path))
    scheduler.observer.observe_current = True
    rows, candidate, snapshot = [], [], {}
    reset_epoch, previous_time = 0, -1.
    unwrapped_yaw, previous_yaw = None, None
    heading_hash = hashlib.sha256((SRC/'sim/low_speed_input.py').read_bytes()).hexdigest()
    header = {'type':'header','scenario':name,'evidence':'PRODUCTION_CLASS_CALLBACK_HEADING_ASSIST_LOCAL_SIM',
              'hardware_verified':False,'normal_startup':True,'initial_forward_limit_m_s':forward_limit,
              'requested_ticks':tick_limit,'events':events,
              'physical_parameters':{'kp':200,'height_m':.12,'timestep_s':.005,'hz':50.,'delays_zero':True},
              'input_source_sha256':heading_hash,
              'policy_sha256':hashlib.sha256((SRC/'agents/walk.onnx').read_bytes()).hexdigest()}
    invariant_failures = []
    with path.open('w',encoding='utf-8') as output:
        output.write(json.dumps(header)+'\n')
        def before(obs):
            nonlocal candidate,snapshot
            candidate=walk.build_observation(obs)
            heading=controller.read_body_heading()
            reference=keyboard._heading_reference
            snapshot={'actual_heading_before_step_rad':heading,
                      'desired_user_yaw_rate_rad_s':keyboard._physical_current['vtheta'],
                      'heading_reference_rad':reference,
                      'heading_error_before_step_rad':wrapped(reference-heading) if reference is not None else None,
                      'heading_feedback_correction_rad_s':keyboard.heading_correction,
                      'selected_forward_limit_m_s':keyboard.FORWARD_LIMIT,
                      'physical_target':dict(keyboard._physical_target),
                      'physical_current':dict(keyboard._physical_current)}
            if source.last_events and source.last_events[0]['keycode']==X:
                if reference is not None or keyboard.heading_correction!=0.:
                    invariant_failures.append({'tick':source.tick-1,'error':'X did not cancel feedback/reference in next read'})
        def record(tick,obs,command):
            nonlocal reset_epoch,previous_time,unwrapped_yaw,previous_yaw
            now=controller.simulation_time
            if now<previous_time:
                reset_epoch+=1
                unwrapped_yaw=None
            previous_time=now
            yaw=controller.read_body_heading()
            unwrapped_yaw=yaw if unwrapped_yaw is None else unwrapped_yaw+wrapped(yaw-previous_yaw)
            previous_yaw=yaw
            down=quat_apply_inverse(obs.robot_state.body_quat,[0.,0.,-1.])
            tilt=math.degrees(math.acos(max(-1.,min(1.,-down[2]))))
            height=float(controller._data.qpos[2])
            row={'type':'tick','tick':tick,'global_time_s':(tick+1)/50.,'simulation_time_s':now,
                 'reset_epoch':reset_epoch,'events':source.last_events,**snapshot,
                 'actual_heading_after_step_rad':yaw,'actual_heading_unwrapped_rad':unwrapped_yaw,
                 'actual_scheduler_command':obs.user_input.velocity,'move_state':walk.state.name,
                 'firmware_kp':controller._bam.model.actuator.kp,'height_m':height,'tilt_deg':tilt,
                 'fallen':height<.1 or tilt>60.,'qpos':controller._data.qpos.tolist(),
                 'qvel':controller._data.qvel.tolist(),
                 'body_velocity':quat_apply_inverse(obs.robot_state.body_quat,controller._data.qvel[:3].tolist()),
                 'body_quat':obs.robot_state.body_quat,'projected_up':obs.robot_state.projected_gravity,
                 'gyro':obs.robot_state.gyro,'policy_observation_candidate':candidate,
                 'last_action':list(walk._last_action),
                 'joint_targets':{k:float(v) for k,v in command.target_angles.items()},
                 'applied_motor_torques_nm':controller._data.actuator_force.tolist(),
                 'simulated_motor_currents_a':obs.robot_state.motor_currents}
            rows.append(row);output.write(json.dumps(row,allow_nan=False)+'\n')
        completed=scheduler.run(max_ticks=tick_limit,realtime=False,
                                time_source=lambda:controller.simulation_time,
                                observation_callback=before,tick_callback=record)
        fall=[r for r in rows if r['fallen']]
        boundaries={'A':[(6.,30.)],
                    'B':[(4.,10.),(13.,25.),(27.,30.),(35.,40.),(44.,55.),(58.,60.),(65.,70.)],
                    'C':[(6.,15.),(17.,20.),(25.,35.)]}[name]
        phases=[]
        for lo,hi in boundaries:
            samples=[r for r in rows if lo<=r['global_time_s']<hi]
            errors=[abs(math.degrees(r['heading_error_before_step_rad'])) for r in samples
                    if r['heading_error_before_step_rad'] is not None]
            phases.append({'from_global_s':lo,'to_global_s':hi,
                           'forward_command_at_end':samples[-1]['actual_scheduler_command']['vx'],
                           'applied_yaw_command_at_end':samples[-1]['actual_scheduler_command']['vtheta'],
                           'mean_body_vx_m_s':sum(r['body_velocity'][0] for r in samples)/len(samples),
                           'max_abs_body_vx_m_s':max(abs(r['body_velocity'][0]) for r in samples),
                           'mean_body_vy_m_s':sum(r['body_velocity'][1] for r in samples)/len(samples),
                           'mean_actual_yaw_rate_rad_s':sum(r['qvel'][5] for r in samples)/len(samples),
                           'mean_applied_yaw_cmd_rad_s':sum(r['actual_scheduler_command']['vtheta'] for r in samples)/len(samples),
                           'mean_heading_feedback_rad_s':sum(r['heading_feedback_correction_rad_s'] for r in samples)/len(samples),
                           'max_heading_error_deg':max(errors) if errors else None,
                           'mean_abs_heading_error_deg':sum(errors)/len(errors) if errors else None})
        summary={'type':'summary','scenario':name,'evidence':header['evidence'],'hardware_verified':False,
                 'completed_ticks':completed,'requested_ticks_completed':completed==tick_limit,
                 'global_elapsed_s':completed/50.,'simulation_time_s':controller.simulation_time,
                 'reset_epochs':reset_epoch,'fallen_ticks':len(fall),
                 'first_fall':{k:fall[0][k] for k in ('global_time_s','simulation_time_s','height_m','tilt_deg','actual_scheduler_command')} if fall else None,
                 'min_height_m':min(r['height_m'] for r in rows),'max_tilt_deg':max(r['tilt_deg'] for r in rows),
                 'final_unwrapped_heading_deg':math.degrees(rows[-1]['actual_heading_unwrapped_rad']),
                 'final_heading_reference_rad':rows[-1]['heading_reference_rad'],
                 'max_total_current_a':max(sum(abs(v) for v in r['simulated_motor_currents_a'].values()) for r in rows),
                 'max_single_current_a':max(abs(v) for r in rows for v in r['simulated_motor_currents_a'].values()),
                 'X_checked_events':sum(bool(r['events']) and r['events'][0]['keycode']==X for r in rows),
                 'X_next_read_feedback_zero_and_reference_none':(not invariant_failures)
                     if any(r['events'] and r['events'][0]['keycode']==X for r in rows) else None,
                 'invariant_failures':invariant_failures,
                 'event_readbacks':[r for r in rows if r['events']],'phase_stats':phases}
        output.write(json.dumps(summary,allow_nan=False)+'\n')
    controller.shutdown()
    summary['log_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (folder/f'production_speed_keyboard_{name}_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert not invariant_failures, invariant_failures
    return {k:v for k,v in summary.items() if k!='event_readbacks'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',required=True,choices=('A','B','C'))
    parser.add_argument('--output-dir',type=Path,default=SRC.parent/'.omx/artifacts/speed-probes-20261010')
    args=parser.parse_args()
    print(json.dumps(run_scenario(args.scenario,args.output_dir),allow_nan=False))
