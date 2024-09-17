#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.signal import iirfilter, sosfilt

from geometry_msgs.msg import TwistStamped, PointStamped
from std_msgs.msg       import Float64, Bool
from events_shaping_controller_msgs.msg import Vector2
from smooth_trajectory_generator import SmoothedConstantCyclicTrajectory

from pathlib import Path
from collections import deque

path = Path(__file__).parent

class CommandHeuristic():
    
    def __init__(self):

        # Publisher created first such that subscriber can get everything published.
        self.pub_vels_ = rospy.Publisher("~vXYandwZ", TwistStamped, queue_size=10)
        self.pub_wXY_ground_truth = rospy.Publisher("~wXY_groundTruth", Vector2, queue_size=10)
        self.pub_jac_vz_mean = rospy.Publisher("~vz_jacobian_mean", Float64, queue_size=10)
        self.pub_error_accumulated_ = rospy.Publisher("~error_accumulated", Float64, queue_size=10)

        try:
            sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("Command Heuristics combined didn't receive sync message")
            rospy.signal_shutdown("Command Heuristics combined didn't receive sync message")

        rospy.loginfo(f"command heuristics starts {rospy.Time.now()}")

        self.camera_frame_id = "" # TODO

        command_rate = rospy.get_param(
            "~command_rate", 1000) # Hz
        self.time_threshold_stop_commanding = rospy.get_param(
            "~time_threshold_stop_commanding_on_no_input", 1/100) #s
        self.gain_heuristic_cycle_control_commands = rospy.get_param(
            "~heuristic_cycle/gain_heuristic_cycle_control_commands", 0.04)
        self.gain_heuristic_perturbation_control_commands = rospy.get_param(
            "~heuristic_perturbation/gain_heuristic_perturbation_control_commands", 0.0001)
        self.gain_vXY_modulation_based_centroids = rospy.get_param(
            "~gain_vxy_modulation_based_centroids", 1.)
        self.jacobian_window_time = rospy.get_param(
            "/derivative_evaluator/window_time", 0.02)

        self.frequency_heuristic_cycle_control_commands = rospy.get_param(
            "~heuristic_cycle/freq_heuristic_cycle_control_commands", 0.5)
        self.frequency_heuristic_perturbation_control_commands = rospy.get_param(
            "~heuristic_perturbation/freq_heuristic_perturbation_control_commands", 5.)

        self.gain_heuristic_approach_control_commands = rospy.get_param(
            "~heuristic_approach/gain_heuristic_approach_control_commands", 0.2)
        self.approach_extragain = rospy.get_param(
            "~heuristic_approach/approach_extra_gain", 1.3)
        self.approach_command_cyle_interval = rospy.get_param(
            "~heuristic_approach/approach_command_cycle_interval", 5) # it specifies after how may cycles the approach command is sent
        self.frequency_heuristic_approach_control_commands = rospy.get_param(
            "~heuristic_approach/freq_heuristic_approach_control_commands", 1.)
        self.current_heuristic_mode_raw = rospy.get_param(
            "~current_heuristic_mode")
        self.groud_truth_distance = rospy.get_param(
            "~ground_truth_distance", 0.489) # on table
        self.sensor_size_xy = np.array([346, 260]) # TODO: MAKE IT PARAMETERIZABLE

        #self.groud_truth_distance = 0.489 - 0.064 # Pattern on the cube
        # self.groud_truth_distance = 0.489  # Pattern on the table
        # self.groud_truth_distance = 0.489 - 0.22  # Pattern on the cylinder

        self.smooth_constant_velocity_generator = SmoothedConstantCyclicTrajectory(self.frequency_heuristic_cycle_control_commands, 0.4)
        self.all_individual_supported_heuristic_modes = {

            # Generate cycle modes
            'cycle': [
                'sinX',
                'sinY',
                'circularCW',
                'circularCCW',
                'smoothConstantX',
                'smoothConstantY'
            ],
            'perturbation': [
                'sinZ'
            ],
            'approach': [
                'sinZ',
                'constant'
            ]
        }

        try:
            self.current_heuristic_mode = self.interpret_topmode_and_submode(self.current_heuristic_mode_raw)
        except RuntimeError as e:
            rospy.logerr(e)
            rospy.signal_shutdown(e)
            return

        rospy.loginfo(f"Current selected heurisitic mode: {self.current_heuristic_mode}")

        # Signal processing setup
        cutoff_frequency_v = 10
        self.lowpass_filter_v_sos = iirfilter(30, cutoff_frequency_v, btype='lowpass', ftype='butter', output='sos', fs=command_rate)
        self.sos_v_zs = np.zeros((self.lowpass_filter_v_sos.shape[0], 2, 2)) # Meant to store delays

        cutoff_frequency_vz = 1
        self.lowpass_filter_vz_sos = iirfilter(30, cutoff_frequency_vz, btype='lowpass', ftype='butter', output='sos', fs=command_rate)
        self.sos_vz_zs = np.zeros((self.lowpass_filter_vz_sos.shape[0], 1, 2)) # Meant to store delays

        self.last_vXY = [deque(maxlen=2), deque(maxlen=2)]
        self.cycle_change_time_t0 = np.array([None, None])
        self.num_zero_crossing_vXY = np.array([0., 0.])
        self.last_centroid_error_mean = np.array([0., 0.])

        self.centroid_error = []
        self.last_centroid_error = None
        self.centroid_timestamps = []

        self.centroid_error_modulation_shift = np.array([0., 0.])

        self.last_vZ = deque(maxlen=2)
        self.num_zero_crossing_down_up_vZ = 0

        self.last_msg_rcvd_time = None
        self.time_in_the_beginning = None

        self.approach_half_cycle = False
        self.jacobian_sharpness_vz_values = []
        self.jacobian_sharpness_vz_ts = []
        self.jacobian_eval_start_ts = None

        self.ids = 0

        # NOTE: This won't be real-time! This only aims to match expected rate
        self.issue_command_callback_timer_ = rospy.Timer(rospy.Duration(1/command_rate), self.issue_command_callback)
        self.count_periods=0

        self.current_accumulate_error = np.array([0., 0.])
        self.sub_event_centroid = rospy.Subscriber("gaze_lock_commander/centroid_error", PointStamped, self.event_centroid_callback)
        self.sub_event_jacobian = rospy.Subscriber("derivative_evaluator/jacobian", PointStamped, self.event_jacobian_callback)

    def event_jacobian_callback(self, msg):

        jac_val = msg.point.z
        time_stamp = msg.header.stamp.to_sec()
        self.jacobian_sharpness_vz_values.append(jac_val)
        self.jacobian_sharpness_vz_ts.append(time_stamp)

    def event_centroid_callback(self, msg):

        self.centroid_error.append([msg.point.x, msg.point.y])
        self.last_centroid_error = np.array([msg.point.x, 0.])
        self.centroid_timestamps.append(rospy.Time.now().to_sec())

    def interpret_topmode_and_submode(self, input_mode_string):

        # Input should be delimited by _ for submode, and + for topmode.
        # For example, topmode1_sub-mode-1+topmode2_sub-mode-2
        # means two top modes, topmode1 and topmode2 with their relatives
        # submodes.

        input_fullmodes_raw = input_mode_string.split('+')
        input_fullmodes = {}

        for fullmode_raw in input_fullmodes_raw:
            topmode_and_submode = fullmode_raw.split('_')
            if len(topmode_and_submode) == 1:
                submode = None
            else:
                submode = topmode_and_submode[1]
            input_fullmodes[topmode_and_submode[0]] = submode

        if len(input_fullmodes) == 0:
            raise RuntimeError('Must specify at least one operating top mode!')

        for input_topmode in input_fullmodes:
            if input_topmode not in self.all_individual_supported_heuristic_modes.keys():
                raise RuntimeError(
                    f"Heuristic top mode {input_topmode} is not supported!"
                    f" Supported top modes = {self.all_individual_supported_heuristic_modes.keys()}")

            submode = input_fullmodes[input_topmode]
            if submode == None:
                input_fullmodes[input_topmode] = None
                continue

            elif submode not in self.all_individual_supported_heuristic_modes[input_topmode]:
                raise RuntimeError(
                    f"Heuristic submode {submode} is not supported!"
                    f" Supported {input_topmode} submodes = {self.all_individual_supported_heuristic_modes[input_topmode]}")

        return input_fullmodes


    def cyclic_movement_modulation_step(self, time_current, axis):

        if self.cycle_change_time_t0[axis] == None:
            self.cycle_change_time_t0[axis] = time_current
            return None

        self.num_zero_crossing_vXY[axis] += 1
        if self.num_zero_crossing_vXY[axis] == 2:
            index_filtered = np.argwhere((np.array(self.centroid_timestamps) > self.cycle_change_time_t0[axis]) & (np.array(self.centroid_timestamps) < time_current))[:, 0]
            if index_filtered.shape[0] < 2:
                return None
            all_errors = np.array(self.centroid_error[index_filtered[0]:index_filtered[-1]])
            self.last_centroid_error_mean = np.mean(all_errors, axis=0)
            msg_ = Float64()
            msg_.data = self.last_centroid_error_mean[axis]
            self.pub_error_accumulated_.publish(msg_)
            self.centroid_error_modulation_shift[axis] = float(self.last_centroid_error_mean[axis]) / float(self.sensor_size_xy[axis]) * self.gain_vXY_modulation_based_centroids
            self.centroid_error = self.centroid_error[index_filtered[0]:index_filtered[-1]]
            self.cycle_change_time_t0[axis] = time_current
            self.centroid_timestamps = self.centroid_timestamps[index_filtered[0]:index_filtered[-1]]
            self.num_zero_crossing_vXY[axis] = 0


    def issue_command_callback(self, timer_obj):

        time_now = rospy.Time().now()
        time_now_secs = time_now.to_sec()
        if self.last_msg_rcvd_time is not None:
            time_since_last_msg = (time_now - self.last_msg_rcvd_time).to_sec()

            if time_since_last_msg > self.time_threshold_stop_commanding:
                rospy.logwarn_throttle(1.0,
                                       f"Heuristic commander stopped issuing commands as time since last input is {time_since_last_msg:.2f} s")
                return

        if self.time_in_the_beginning is None:
            self.time_in_the_beginning = time_now
        time_elapsed_since_beginning = (time_now - self.time_in_the_beginning)

        vXY_heuristics = np.array([0., 0.])
        vZ_perturbation = np.array([0.])
        vZ_approach = np.array([0.])
        for topmode in self.current_heuristic_mode:
            submode = self.current_heuristic_mode[topmode]

            if topmode == 'cycle':
                sinusoidal = np.sin(2 * np.pi * time_elapsed_since_beginning.to_sec() * self.frequency_heuristic_cycle_control_commands)
                cosinusoidal = np.cos(2 * np.pi * time_elapsed_since_beginning.to_sec() * self.frequency_heuristic_cycle_control_commands)
                smooth_constant = self.smooth_constant_velocity_generator.signal_at(time_elapsed_since_beginning.to_sec())
                if submode == 'sinX':
                    vXY_heuristics = np.array([sinusoidal, 0.])
                elif submode == 'sinY':
                    vXY_heuristics = np.array([0., sinusoidal])
                elif submode == 'circularCCW':
                    vXY_heuristics = np.array([cosinusoidal, sinusoidal])
                elif submode == 'circularCW':
                    vXY_heuristics = np.array([sinusoidal, cosinusoidal])
                elif submode == 'smoothConstantX':
                    vXY_heuristics = np.array([smooth_constant, 0])
                elif submode == 'smoothConstantY':
                    vXY_heuristics = np.array([0., smooth_constant])
                else:
                    raise RuntimeError

                self.last_vXY[0].append(vXY_heuristics[0])
                self.last_vXY[1].append(vXY_heuristics[1])

                for i, deque_i in enumerate(self.last_vXY):
                    if len(deque_i) == 2:
                        sign_commands = np.sign(deque_i)
                        if (sign_commands[0] >= 0 and sign_commands[1] < 0) or (
                                sign_commands[0] < 0 and sign_commands[1] >= 0):
                            current_time = rospy.Time.now().to_sec()
                            self.cyclic_movement_modulation_step(current_time, i)

            if topmode == 'perturbation':
                if submode == 'sinZ':
                    vZ_perturbation[0] = np.sin(2 * np.pi * time_elapsed_since_beginning.to_sec() * self.frequency_heuristic_perturbation_control_commands)
                else:
                    RuntimeError
            if topmode == 'approach':
                if submode == 'sinZ':
                    vZ_approach[0] = np.sin(2 *np.pi * time_elapsed_since_beginning.to_sec() * self.frequency_heuristic_approach_control_commands)
                    self.last_vZ.append(vZ_approach[0])
                    if len(self.last_vZ) == 2:
                        sign_commands = np.sign(self.last_vZ)
                        if sign_commands[0] < 0 and sign_commands[1] >= 0:
                            self.num_zero_crossing_down_up_vZ += 1
                    if self.num_zero_crossing_down_up_vZ % self.approach_command_cyle_interval == 0 and self.num_zero_crossing_down_up_vZ != 0 and \
                            vZ_approach[0] > 0:
                        if self.approach_half_cycle is False and self.jacobian_eval_start_ts is not None:
                            time_stamps_array = np.array(self.jacobian_sharpness_vz_ts)
                            idx = np.argwhere(
                                (time_stamps_array - self.jacobian_window_time > self.jacobian_eval_start_ts) & (
                                        time_stamps_array < time_now_secs))
                            vz_jacobian_array = np.array(self.jacobian_sharpness_vz_values[np.min(idx): np.max(idx)])
                            self.jacobian_sharpness_vz_ts = self.jacobian_sharpness_vz_ts[np.min(idx):]
                            self.jacobian_sharpness_vz_values = self.jacobian_sharpness_vz_values[np.min(idx):]
                            msg_jac = Float64()
                            # np.save(path / f"data_{self.ids}.npy", vz_jacobian_array)
                            self.ids += 1
                            data_filtered = vz_jacobian_array
                            msg_jac.data = np.max(np.abs(data_filtered))
                            self.pub_jac_vz_mean.publish(msg_jac)
                            rospy.logerr(time_now_secs - self.jacobian_eval_start_ts)

                        vZ_approach *= self.approach_extragain
                        self.approach_half_cycle = True

                    if self.approach_half_cycle and vZ_approach[0] < 0:
                        # ANALYSIS STARTING POINT
                        self.jacobian_eval_start_ts = time_now_secs
                        self.approach_half_cycle = False
                elif submode == 'constant':
                    vZ_approach[0] = 1
                else:
                    raise RuntimeError

        vXY_heuristics += self.centroid_error_modulation_shift
        vXY_heuristics_with_gain = self.gain_heuristic_cycle_control_commands * vXY_heuristics
        vZ_heuristics = (self.gain_heuristic_perturbation_control_commands * vZ_perturbation) + (self.gain_heuristic_approach_control_commands * vZ_approach)
        # Signal processing
        """vXY_filtered, self.sos_v_zs = sosfilt(self.lowpass_filter_v_sos, vXY_unfiltered.reshape(-1, 1), axis=-1, zi=self.sos_v_zs)
        wZ_filtered, self.sos_vz_zs = sosfilt(self.lowpass_filter_vz_sos, wZ_unfiltered.reshape(-1, 1), axis=-1, zi=self.sos_vz_zs)
        wXY_filtered, self.sos_w_zs = sosfilt(self.low_pass_filter_w_sos, wXY_unfiltered.reshape(-1, 1), axis=-1, zi=self.sos_w_zs)"""

        vZ_saturated = np.clip(vZ_heuristics, -0.15, 0.15)
        vXY_saturated = np.clip(vXY_heuristics_with_gain, -0.15, 0.15) #TODO


        pub_msg = TwistStamped()
        pub_msg.header.stamp = time_now
        pub_msg.header.frame_id = self.camera_frame_id

        pub_msg.twist.linear.x = vXY_saturated[0]
        pub_msg.twist.linear.y = vXY_saturated[1]
        pub_msg.twist.linear.z = vZ_saturated[0]

        # NaNs for dimensions not commanded
        pub_msg.twist.angular.x = np.nan
        pub_msg.twist.angular.y = np.nan
        pub_msg.twist.angular.z = 0.

        msg_wXY_gt = Vector2()
        msg_wXY_gt.x = (1 / self.groud_truth_distance) * vXY_saturated[1]
        msg_wXY_gt.y = - (1 / self.groud_truth_distance) * vXY_saturated[0]

        self.pub_vels_.publish(pub_msg)
        self.pub_wXY_ground_truth.publish(msg_wXY_gt)

if __name__ == "__main__":
    rospy.init_node('heurisitic_commander')
    CommandHeuristic()
    rospy.spin()
