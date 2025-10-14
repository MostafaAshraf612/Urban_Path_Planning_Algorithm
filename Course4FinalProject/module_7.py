#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import warnings
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser
import local_planner
import behavioural_planner

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils

# ---------------------------------------------------------------------------
# Suppress specific matplotlib tight_layout / Agg fallback warnings so the
# terminal stays readable during runs. This is safe: it only hides user
# warnings related to tight_layout behavior, not errors.
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout.*")
# ---------------------------------------------------------------------------

"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 100.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 2      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]

PLAYER_START_INDEX = 1
FIGSIZE_X_INCHES   = 15
FIGSIZE_Y_INCHES   = 10
PLOT_LEFT          = 0.1
PLOT_BOT           = 0.1
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

WAYPOINTS_FILENAME = 'course4_waypoints.txt'
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0

# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 8.0
BP_LOOKAHEAD_TIME      = 2.0
PATH_OFFSET            = 1.5
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0]
CIRCLE_RADII           = [1.5, 1.5, 1.5]
TIME_GAP               = 1.0
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 1.5
SLOW_SPEED             = 2.0
STOP_LINE_BUFFER       = 3.5
LEAD_VEHICLE_LOOKAHEAD = 20.0
LP_FREQUENCY_DIVISOR   = 2

C4_STOP_SIGN_FILE        = 'stop_sign_params.txt'
C4_STOP_SIGN_FENCELENGTH = 5
C4_PARKED_CAR_FILE       = 'parked_vehicle_params.txt'

INTERP_MAX_POINTS_PLOT    = 10
INTERP_DISTANCE_RES       = 0.01

CONTROLLER_OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'controller_output')

def make_carla_settings(args):
    settings = CarlaSettings()
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings

class Timer(object):
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        return self.elapsed_seconds_since_lap() >= self._period_for_lap

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)
    return (x, y, yaw)

def get_start_pos(scene):
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)
    return (x, y, yaw)

def get_player_collided_flag(measurement,
                             prev_collision_vehicles,
                             prev_collision_pedestrians,
                             prev_collision_other):
    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

def send_control_command(client, throttle, steer, brake,
                         hand_brake=False, reverse=False):
    control = VehicleControl()
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    # robust directory creation
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception:
        if not os.path.exists(output_folder):
            raise

def store_trajectory_plot(graph, fname):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    try:
        graph.savefig(file_name)
    except Exception as e:
        # don't crash on save; warn and continue
        logging.warning("Could not save figure %s: %s", fname, str(e))

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')
    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')
    with open(file_name, 'w') as collision_file:
        collision_file.write(str(sum(collided_list)))
    # Also write 'collision.txt' for scripts that expect this filename
    alt_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision.txt')
    try:
        with open(alt_file, 'w') as f:
            f.write(str(sum(collided_list)))
    except Exception as e:
        logging.warning("Could not write collision.txt: %s", str(e))

def exec_waypoint_nav_demo(args):
    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings = make_carla_settings(args)

        scene = client.load_settings(settings)
        player_start = PLAYER_START_INDEX
        client.start_episode(player_start)

        time.sleep(CLIENT_WAIT_TIME)

        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        # Load config
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))
        demo_opt = config['Demo Parameters']

        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        live_plot_timer = Timer(live_plot_period)

        # Load stop sign params
        stopsign_data = None
        stopsign_fences = []
        with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
            next(stopsign_file)
            stopsign_reader = csv.reader(stopsign_file,
                                         delimiter=',',
                                         quoting=csv.QUOTE_NONNUMERIC)
            stopsign_data = list(stopsign_reader)
            for i in range(len(stopsign_data)):
                stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0

        for i in range(len(stopsign_data)):
            x = stopsign_data[i][0]
            y = stopsign_data[i][1]
            z = stopsign_data[i][2]
            yaw = stopsign_data[i][3] + np.pi / 2.0
            spos = np.array([[0, 0], [0, C4_STOP_SIGN_FENCELENGTH]])
            rotyaw = np.array([[np.cos(yaw), np.sin(yaw)],
                               [-np.sin(yaw), np.cos(yaw)]])
            spos_shift = np.array([[x, x], [y, y]])
            spos = np.add(np.matmul(rotyaw, spos), spos_shift)
            stopsign_fences.append([spos[0,0], spos[1,0], spos[0,1], spos[1,1]])

        # Parked car(s)
        parkedcar_data = None
        parkedcar_box_pts = []
        with open(C4_PARKED_CAR_FILE, 'r') as parkedcar_file:
            next(parkedcar_file)
            parkedcar_reader = csv.reader(parkedcar_file,
                                          delimiter=',',
                                          quoting=csv.QUOTE_NONNUMERIC)
            parkedcar_data = list(parkedcar_reader)
            for i in range(len(parkedcar_data)):
                parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0

        for i in range(len(parkedcar_data)):
            x = parkedcar_data[i][0]
            y = parkedcar_data[i][1]
            yaw = parkedcar_data[i][3]
            xrad = parkedcar_data[i][4]
            yrad = parkedcar_data[i][5]
            cpos = np.array([
                    [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0],
                    [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
            rotyaw = np.array([[np.cos(yaw), np.sin(yaw)],
                               [-np.sin(yaw), np.cos(yaw)]])
            cpos_shift = np.array([[x]*8, [y]*8])
            cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
            for j in range(cpos.shape[1]):
                parkedcar_box_pts.append([cpos[0,j], cpos[1,j]])

        # Load waypoints
        waypoints_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          WAYPOINTS_FILENAME)
        with open(waypoints_filepath) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle,
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)

        # Controller
        controller = controller2d.Controller2D(waypoints)

        # Determine sim timestep
        num_iterations = max(1, ITER_FOR_SIM_TIMESTEP)
        measurement_data, sensor_data = client.read_data()
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        sim_duration = 0
        for i in range(num_iterations):
            measurement_data, sensor_data = client.read_data()
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 - sim_start_stamp

        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        # Frame init
        measurement_data, sensor_data = client.read_data()
        start_timestamp = measurement_data.game_timestamp / 1000.0
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history = [start_x]
        y_history = [start_y]
        yaw_history = [start_yaw]
        time_history = [0]
        speed_history = [0]
        collided_flag_history = [False]

        # Live plotting setup
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

        # trajectory figure
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        trajectory_fig.set_invert_x_axis()
        trajectory_fig.set_axis_equal()

        trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                 x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                                 linestyle="-", marker="", color='g')
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES,
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        trajectory_fig.add_graph("start_pos", window_size=1,
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0],
                                 markertext="Start", marker_text_offset=1)
        trajectory_fig.add_graph("end_pos", window_size=1,
                                 x0=[waypoints_np[-1, 0]],
                                 y0=[waypoints_np[-1, 1]],
                                 marker="D", color='r',
                                 markertext="End", marker_text_offset=1)
        trajectory_fig.add_graph("car", window_size=1,
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        trajectory_fig.add_graph("leadcar", window_size=1,
                                 marker="s", color='g', markertext="Lead Car",
                                 marker_text_offset=1)

        # stopsign plotting: guard in case stopsign_fences empty
        if len(stopsign_fences) > 0:
            trajectory_fig.add_graph("stopsign", window_size=1,
                                     x0=[stopsign_fences[0][0]], y0=[stopsign_fences[0][1]],
                                     marker="H", color="r",
                                     markertext="Stop Sign", marker_text_offset=1)
            trajectory_fig.add_graph("stopsign_fence", window_size=1,
                                     x0=[stopsign_fences[0][0], stopsign_fences[0][2]],
                                     y0=[stopsign_fences[0][1], stopsign_fences[0][3]],
                                     color="r")

        parkedcar_box_pts_np = np.array(parkedcar_box_pts)
        if parkedcar_box_pts_np.size > 0:
            trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                                     x0=parkedcar_box_pts_np[:,0], y0=parkedcar_box_pts_np[:,1],
                                     linestyle="", marker="+", color='b')

        trajectory_fig.add_graph("selected_path",
                                 window_size=INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x]*INTERP_MAX_POINTS_PLOT,
                                 y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                 color=[1, 0.5, 0.0],
                                 linewidth=3)

        for i in range(NUM_PATHS):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                     x0=None, y0=None, color=[0.0, 0.0, 1.0])

        # 1D figures
        forward_speed_fig = lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed",
                                    label="forward_speed",
                                    )
        forward_speed_fig.add_graph("reference_signal",
                                    label="reference_Signal",
                                    )

        throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        throttle_fig.add_graph("throttle",
                               label="throttle",
                               window_size=12)
        brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        brake_fig.add_graph("brake",
                            label="brake",
                            )
        steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        steer_fig.add_graph("steer",
                            label="steer",
                            )

        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()

        # Local & behavioural planners
        wp_goal_index = 0
        local_waypoints = None
        path_validity = np.zeros((NUM_PATHS, 1), dtype=bool)
        lp = local_planner.LocalPlanner(NUM_PATHS,
                                        PATH_OFFSET,
                                        CIRCLE_OFFSETS,
                                        CIRCLE_RADII,
                                        PATH_SELECT_WEIGHT,
                                        TIME_GAP,
                                        A_MAX,
                                        SLOW_SPEED,
                                        STOP_LINE_BUFFER)
        bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                    stopsign_fences,
                                                    LEAD_VEHICLE_LOOKAHEAD)

        reached_the_end = False
        skip_first_frame = True
        current_timestamp = start_timestamp

        prev_collision_vehicles = 0
        prev_collision_pedestrians = 0
        prev_collision_other = 0

        for frame in range(TOTAL_EPISODE_FRAMES):
            measurement_data, sensor_data = client.read_data()

            prev_timestamp = current_timestamp
            current_x, current_y, current_yaw = get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0

            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START

            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp)

            collided_flag, \
            prev_collision_vehicles, \
            prev_collision_pedestrians, \
            prev_collision_other = get_player_collided_flag(measurement_data,
                                                           prev_collision_vehicles,
                                                           prev_collision_pedestrians,
                                                           prev_collision_other)
            collided_flag_history.append(collided_flag)

            # Obtain Lead Vehicle information (safe)
            lead_car_pos = []
            lead_car_length = []
            lead_car_speed = []
            for agent in measurement_data.non_player_agents:
                if agent.HasField('vehicle'):
                    lead_car_pos.append([agent.vehicle.transform.location.x,
                                         agent.vehicle.transform.location.y])
                    lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                    lead_car_speed.append(agent.vehicle.forward_speed)

            # Compute open loop speed & ego_state
            if frame % LP_FREQUENCY_DIVISOR == 0:
                open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)
                ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

                # safe call: only pass lead car if it exists (index 1)
                lead_for_check = None
                if len(lead_car_pos) > 1:
                    lead_for_check = lead_car_pos[1]
                # call transition_state (may rely on lead info separately)
                bp.transition_state(waypoints, ego_state, current_speed)

                if lead_for_check is not None:
                    bp.check_for_lead_vehicle(ego_state, lead_for_check)
                # else skip lead vehicle follow logic

                goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)
                paths, path_validity = lp.plan_paths(goal_state_set)
                paths = local_planner.transform_paths(paths, ego_state)
                collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts])

                best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                if best_index is None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_index]
                    lp._prev_best_path = best_path

                desired_speed = bp._goal_state[2]
                # safe leader state
                if len(lead_car_pos) > 1 and len(lead_car_speed) > 1:
                    lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
                else:
                    lead_car_state = [0.0, 0.0, 0.0]

                decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, bp._follow_lead_vehicle)

                if local_waypoints is not None:
                    wp_distance = []
                    local_waypoints_np = np.array(local_waypoints)
                    for i in range(1, local_waypoints_np.shape[0]):
                        wp_distance.append(np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                                  (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                    wp_distance.append(0)

                    wp_interp = []
                    for i in range(local_waypoints_np.shape[0] - 1):
                        wp_interp.append(list(local_waypoints_np[i]))
                        num_pts_to_interp = max(0, int(np.floor(wp_distance[i] / float(INTERP_DISTANCE_RES)) - 1))
                        wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                        norm = np.linalg.norm(wp_vector[0:2])
                        if norm == 0:
                            continue
                        wp_uvector = wp_vector / norm
                        for j in range(num_pts_to_interp):
                            next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                            wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                    wp_interp.append(list(local_waypoints_np[-1]))
                    controller.update_waypoints(wp_interp)
                else:
                    wp_interp = [[start_x, start_y, 0.0]]  # fallback

            # Controller update
            if local_waypoints is not None and local_waypoints != []:
                controller.update_values(current_x, current_y, current_yaw,
                                         current_speed,
                                         current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            if skip_first_frame and frame == 0:
                pass
            elif local_waypoints is None:
                pass
            else:
                trajectory_fig.roll("trajectory", current_x, current_y)
                trajectory_fig.roll("car", current_x, current_y)
                if len(lead_car_pos) > 1:
                    trajectory_fig.roll("leadcar", lead_car_pos[1][0], lead_car_pos[1][1])
                forward_speed_fig.roll("forward_speed",
                                       current_timestamp,
                                       current_speed)
                forward_speed_fig.roll("reference_signal",
                                       current_timestamp,
                                       controller._desired_speed)
                throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                brake_fig.roll("brake", current_timestamp, cmd_brake)
                steer_fig.roll("steer", current_timestamp, cmd_steer)

                if frame % LP_FREQUENCY_DIVISOR == 0:
                    path_counter = 0
                    for i in range(NUM_PATHS):
                        if path_validity[i]:
                            if not collision_check_array[path_counter]:
                                colour = 'r'
                            elif i == best_index:
                                colour = 'k'
                            else:
                                colour = 'b'
                            trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                            path_counter += 1
                        else:
                            trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')

                # selected path plotting - safe guard for missing wp_interp
                try:
                    wp_interp_np = np.array(wp_interp)
                    path_indices = np.floor(np.linspace(0, wp_interp_np.shape[0]-1, INTERP_MAX_POINTS_PLOT))
                    trajectory_fig.update("selected_path",
                                          wp_interp_np[path_indices.astype(int), 0],
                                          wp_interp_np[path_indices.astype(int), 1],
                                          new_colour=[1, 0.5, 0.0])
                except Exception:
                    pass

                if enable_live_plot and live_plot_timer.has_exceeded_lap_period():
                    lp_traj.refresh()
                    lp_1d.refresh()
                    live_plot_timer.lap()

            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        # End of demo
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")

        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)

        # Save figures and outputs (safe)
        try:
            store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        except Exception:
            pass
        try:
            store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        except Exception:
            pass
        try:
            store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        except Exception:
            pass
        try:
            store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        except Exception:
            pass
        try:
            store_trajectory_plot(steer_fig.fig, 'steer_output.png')
        except Exception:
            pass

        write_trajectory_file(x_history, y_history, speed_history, time_history,
                              collided_flag_history)
        write_collisioncount_file(collided_flag_history)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host server (default: localhost)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('-q', '--quality-level', choices=['Low', 'Epic'], type=lambda s: s.title(), default='Low', help='graphics quality level.')
    argparser.add_argument('-c', '--carla-settings', metavar='PATH', dest='settings_filepath', default=None, help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
