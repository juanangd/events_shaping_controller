#!/usr/bin/env python3

import rospy
import numpy as np
from events_buffering import EventsBuffering
import torch
from geometry_msgs.msg import TwistStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from dvs_msgs.msg import EventArray, Event, EventArrayFlattened
from radial_distortion_filter import RadialDistortionRemover
from smooth_trajectory_generator import SmoothConstantTrajectory
from std_msgs.msg import Bool, Float64
from image_warped_events_evaluator import ImageWarpedEventsEvaluator


class ApproachCommandConstraintDivergence():

	def __init__(self):

		try:
			camera_info = rospy.wait_for_message("/dvs/camera_info", CameraInfo, timeout=50.0)
			self.camera_intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
			self.distortion_param = np.array(camera_info.D)
			self.camera_intrinsic_matrix_inverse = np.linalg.inv(self.camera_intrinsic_matrix)
			self.sensor_size = (camera_info.height, camera_info.width)

		except rospy.ROSException:
			rospy.logerr("CommandGazeLock not receive event-based camera string! Exiting.")
			rospy.signal_shutdown("CommandGazeLock could not receive camera stream!")
			return

		self.camera_intrinsic_matrix = torch.Tensor(self.camera_intrinsic_matrix)
		self.camera_intrinsic_matrix_inverse = torch.Tensor(self.camera_intrinsic_matrix_inverse)
		self.sensor_size = torch.Size(self.sensor_size)

		self.pub_approach_commands_ = rospy.Publisher("~vZ", TwistStamped, queue_size=10)
		self.pub_jacobian_computed_ = rospy.Publisher("~jacobian_computed", PointStamped, queue_size=10)
		self.analyzed_event_pub = rospy.Publisher("~analyzed_events", EventArrayFlattened, queue_size=10)
		self.pub_distance_estimated = rospy.Publisher("~distance_expected", Float64, queue_size=10)
		self.pub_divergence_estimated = rospy.Publisher("~divergence_estimated", Float64, queue_size=10)

		try:
			sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
		except rospy.ROSException:
			rospy.logerr("CommandGazeLock didn't receive sync message")
			rospy.signal_shutdown("CommandGazeLock didn't receive sync message")

		rospy.loginfo(f"command gaze lock starts {rospy.Time.now()}")

		self.bounding_box_height = rospy.get_param(
			"bounding_box/bounding_box_height", 100
		)

		self.publish_events_analyzed = rospy.get_param(
			"~publish_events_analyzed", False)

		self.bounding_box_width = rospy.get_param(
			"bounding_box/bounding_box_width", 250
		)

		self.bounding_box_center_x = rospy.get_param(
			"bounding_box/bounding_box_center_x", self.sensor_size[1] // 2
		)

		self.publish_jacobian_computed = rospy.get_param(
			"~publish_jacobian_computed", True)

		self.bounding_box_center_y = rospy.get_param(
			"bounding_box/bounding_box_center_y", self.sensor_size[0] // 2
		)


		self.analyze_only_bounding_box = rospy.get_param(
			"~analyze_only_bounding_box", True
		)

		self.num_event_threshold = rospy.get_param(
			"~num_events_threshold", 150
		)

		self.sliding_window_time_jacob = rospy.get_param(
			"~sliding_window_time_jacobian", 0.020  # s
		)

		self.maximum_num_events_to_process = rospy.get_param(
			"~maximum_num_events_to_process", 100000  # s
		)

		self.upper_limit_x = self.bounding_box_center_x + (self.bounding_box_width // 2)
		self.upper_limit_y = self.bounding_box_center_y + (self.bounding_box_height // 2)
		self.lower_limit_x = self.bounding_box_center_x - (self.bounding_box_width // 2)
		self.lower_limit_y = self.bounding_box_center_y - (self.bounding_box_height // 2)

		self.alpha_ema = rospy.get_param(
			"~alpha_ema", 0.8)
		self.jacobian_computation_freq = rospy.get_param(
			"~jacobian_computation_freq", 50)
		self.command_rate = rospy.get_param(
			"~command_rate", 1000)  # Hz
		self.time_threshold_stop_commanding = rospy.get_param(
			"~time_threshold_stop_commanding_on_no_input", 1.)  # s
		self.learning_rate = rospy.get_param(
			"~learning_rate", 100.)
		self.max_commander_vel_saturation = rospy.get_param(
			"~max_commander_vel_saturation", 0.3)  # m / s
		self.sharpness_function_type = rospy.get_param(
			"~sharpness_function_type", "variance")  # rad / s
		self.velocity_gain = rospy.get_param(
			"~velocity_gain", 0.07)
		self.camera_frame_id = ""
		self.target_divergence_rate = rospy.get_param(
			"~divergence_rate_target", -0.25)
		self.init_smoothing_time = rospy.get_param(
			"~init_smoothing_time", 0.5)

		self.initial_distance = rospy.get_param(
			"~initial_measured_distance", 0.7)

		self.current_expected_distance_from_distance = self.initial_distance
		self.current_divergence_estimated_from_distance = None
		self.current_velocity = None
		self.radial_distortion_remover = RadialDistortionRemover(fx=float(self.camera_intrinsic_matrix[0, 0]),
																 fy=float(self.camera_intrinsic_matrix[1, 1]),
																 cx=float(self.camera_intrinsic_matrix[0, 2]),
																 cy=float(self.camera_intrinsic_matrix[1, 2]),
																 k1=self.distortion_param[0],
																 k2=self.distortion_param[1],
																 t1=self.distortion_param[2],
																 t2=self.distortion_param[3],
																 k3=self.distortion_param[4])

		# self.optimizer = CmaxOptimizer(self.camera_intrinsic_matrix, self.camera_intrinsic_matrix_inverse, self.sensor_size, 11, torch.Tensor([12, 12]), sharpness_function_type="poisson")
		self.iwe_evaluator = ImageWarpedEventsEvaluator(self.camera_intrinsic_matrix,
														self.camera_intrinsic_matrix_inverse, self.sensor_size, 11,
														torch.Tensor([2., 2.]),
														sharpness_function_type=self.sharpness_function_type, motion_model="translation_divergence")

		self.smooth_trajectory_generator = SmoothConstantTrajectory(smoothing_time=self.init_smoothing_time)

		self.event_buffering = EventsBuffering(use_event_time=True)

		# Signal processing setup
		"""cutoff_frequency = 10
		self.lowpass_filter_sos = iirfilter(2, cutoff_frequency, rs=15, btype='lowpass', ftype='cheby2', output='sos',
											fs=self.command_rate)
		self.sos_zs = np.zeros((self.lowpass_filter_sos.shape[0], 1, 2))  # Meant to store delays"""
		self.last_msg_received = None

		# Structures needed
		self.last_jacobian_value = torch.tensor(0.)
		self.starting_time = None
		self.start_analysing_data = False

		# Ros Subscriber / timers etc
		rospy.Timer(rospy.Duration(1 / self.command_rate), self.issue_command_callback)
		rospy.Subscriber("/dvs/events_flattened", EventArrayFlattened, self.events_in_callback)
		rospy.Timer(rospy.Duration(1/self.jacobian_computation_freq), self.jacobian_computation)
		rospy.Timer(rospy.Duration(1/20), self.update_expected_divergence)


	def update_expected_divergence(self, time_st):

		if self.current_divergence_estimated_from_distance is not None:
			msg_div = Float64()
			msg_div.data= self.current_divergence_estimated_from_distance
			self.pub_divergence_estimated.publish(msg_div)

		msg_dist = Float64()
		msg_dist.data = self.current_expected_distance_from_distance
		self.pub_distance_estimated.publish(msg_dist)

	def events_in_callback(self, msg):

		self.event_buffering.events_callback(msg)

	def jacobian_computation(self, time_st):

		num_events  = 0
		if self.start_analysing_data:
			data = self.event_buffering.pull_data(self.sliding_window_time_jacob, remove_old_packets=True)
			data = self.radial_distortion_remover.undistort_events(data)
			self.last_msg_received = rospy.Time.now()
			jac = torch.tensor(0.)

			if data is not None:
				self.current_number_events = data.shape[0]
				array_tensor = torch.Tensor(data)
				if self.analyze_only_bounding_box:
					valid_idx = torch.where(array_tensor[:, 0] > self.lower_limit_x, torch.Tensor([1]), torch.Tensor([0]))
					valid_idx *= torch.where(array_tensor[:, 0] < self.upper_limit_x, torch.Tensor([1]), torch.Tensor([0]))
					valid_idx *= torch.where(array_tensor[:, 1] < self.upper_limit_y, torch.Tensor([1]), torch.Tensor([0]))
					valid_idx *= torch.where(array_tensor[:, 1] > self.lower_limit_y, torch.Tensor([1]), torch.Tensor([0]))

					array_tensor = array_tensor[valid_idx.bool(), :]
				num_events = array_tensor.shape[0]
				if array_tensor.shape[0] > self.num_event_threshold:
					if array_tensor.shape[0] > self.maximum_num_events_to_process: # DOWNSAMPLE IT
						array_tensor[::array_tensor.shape[0]//self.maximum_num_events_to_process]
					jac = self.iwe_evaluator.jacobian_loss_fn(torch.tensor(self.target_divergence_rate), array_tensor)

				if self.publish_events_analyzed:
					events_array = EventArrayFlattened()
					events_array.header.stamp = self.last_msg_received
					events_array.events = array_tensor.flatten().tolist()
					self.analyzed_event_pub.publish(events_array)
			jac_to_use = jac if jac > 0 else 0
			self.last_jacobian_value = (1 - self.alpha_ema) * self.last_jacobian_value + self.alpha_ema * jac_to_use

			self.current_velocity -= self.learning_rate * self.last_jacobian_value * self.current_velocity
			if self.publish_jacobian_computed:
				point = PointStamped()
				point.point.x = jac
				point.point.y = num_events
				self.pub_jacobian_computed_.publish(point)

	def issue_command_callback(self, timer_obj: rospy.timer.TimerEvent):

		current_time = rospy.Time.now()
		if self.starting_time is None:
			self.starting_time = current_time

		if self.current_velocity is not None:
			self.current_expected_distance_from_distance -= self.current_velocity * (timer_obj.current_real - timer_obj.last_real).to_sec()
			self.current_divergence_estimated_from_distance = self.current_velocity / self.current_expected_distance_from_distance

		current_relative_time = (current_time - self.starting_time).to_sec()
		if current_relative_time < self.init_smoothing_time + self.sliding_window_time_jacob:
			self.current_velocity = self.velocity_gain * self.smooth_trajectory_generator.signal_at(current_relative_time)
			self.last_msg_received = current_time

		elif not self.start_analysing_data:
			self.current_velocity = self.velocity_gain
			self.start_analysing_data = True

		time_since_last_msg = (current_time - self.last_msg_received).to_sec()
		if time_since_last_msg > self.time_threshold_stop_commanding:
			return None

		wXY_wo_gain_unfiltered = np.array([self.current_velocity])

		"""wXY_wo_gain_filtered, self.sos_zs = sosfilt(self.lowpass_filter_sos, wXY_wo_gain_unfiltered.reshape(-1, 1),
													axis=-1, zi=self.sos_zs)"""

		vz_filtered = wXY_wo_gain_unfiltered

		pub_msg = TwistStamped()
		pub_msg.header.stamp = current_time
		pub_msg.header.frame_id = self.camera_frame_id

		# Saturate velocities at the command level
		vz_filtered_saturated = np.clip(vz_filtered, -self.max_commander_vel_saturation, self.max_commander_vel_saturation)

		pub_msg.twist.angular.x = np.nan
		pub_msg.twist.angular.y = np.nan
		pub_msg.twist.angular.z = np.nan

		# NaNs for dimensions not commanded
		pub_msg.twist.linear.x = np.nan
		pub_msg.twist.linear.y = np.nan
		pub_msg.twist.linear.z = vz_filtered_saturated[0]

		self.pub_approach_commands_.publish(pub_msg)


if __name__ == "__main__":
	rospy.init_node('approach_commander')
	ApproachCommandConstraintDivergence()
	rospy.spin()
