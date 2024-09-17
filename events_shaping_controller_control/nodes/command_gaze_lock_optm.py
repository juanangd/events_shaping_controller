#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from events_buffering import EventsBuffering
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from dvs_msgs.msg import EventArray, Event, EventArrayFlattened
from events_shaping_controller_msgs.msg import Vector2Stamped
from std_msgs.msg import Bool
from image_warped_events.image_warped_events_evaluator import ImageWarpedEventsEvaluator
from events_centroid_tracking import EventsCentroidTracker


class CommandGazeLock():

	def __init__(self):

		try:
			camera_info = rospy.wait_for_message("/dvs/camera_info", CameraInfo, timeout=50.0)
			self.camera_intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
			self.camera_intrinsic_matrix_inverse = np.linalg.inv(self.camera_intrinsic_matrix)
			self.sensor_size = (camera_info.height, camera_info.width)

		except rospy.ROSException:
			rospy.logerr("CommandGazeLock not receive event-based camera string! Exiting.")
			rospy.signal_shutdown("CommandGazeLock could not receive camera stream!")
			return

		self.camera_intrinsic_matrix = torch.Tensor(self.camera_intrinsic_matrix)
		self.camera_intrinsic_matrix_inverse = torch.Tensor(self.camera_intrinsic_matrix_inverse)
		self.sensor_size = torch.Size(self.sensor_size)

		self.bridge = CvBridge()
		self.pub_angular_vels_ = rospy.Publisher("~wXY", TwistStamped, queue_size=10)
		self.pub_centroid_error = rospy.Publisher("~centroid_error", PointStamped, queue_size=10)
		self.pub_image = rospy.Publisher("~image_ae", Image, queue_size=10)
		self.raw_velocity_pub = rospy.Publisher("~jacobian_computation", PointStamped, queue_size=10)
		self.analyzed_event_pub = rospy.Publisher("~analyzed_events", EventArrayFlattened, queue_size=10)

		try:
			sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
		except rospy.ROSException:
			rospy.logerr("CommandGazeLock didn't receive sync message")
			rospy.signal_shutdown("CommandGazeLock didn't receive sync message")

		rospy.loginfo(f"command gaze lock starts {rospy.Time.now()}")

		self.bounding_box_height = rospy.get_param(
			"bounding_box/bounding_box_height", 100
		)

		self.bounding_box_width = rospy.get_param(
			"bounding_box/bounding_box_width", 250
		)

		self.bounding_box_center_x = rospy.get_param(
			"bounding_box/bounding_box_center_x", self.sensor_size[1] // 2
		)

		self.bounding_box_center_y = rospy.get_param(
			"bounding_box/bounding_box_center_y", self.sensor_size[0] // 2
		)

		self.params_to_evaluate = rospy.get_param(
			"~params_to_evaluate", [False, True, False]
		)

		self.analyze_only_bounding_box = rospy.get_param(
			"~analyze_only_bounding_box", True
		)

		self.is_centroid_rbg_based = rospy.get_param(
			"~is_centroid_rbg_based", False)

		self.num_events_threshold = rospy.get_param(
			"~num_events_threshold", 300
		)

		self.sliding_window_time_jacob = rospy.get_param(
			"~sliding_window_time_jacobian", 0.020  # s
		)

		self.sliding_window_time_centroid = rospy.get_param(
			"~sliding_window_time_centroid", 0.02 # s
		)

		self.motion_model = rospy.get_param(
			"~motion_model", "rotation"
		)

		self.jacobian_clipping_value = rospy.get_param(
			"~jacobian_clipping_value", 20
		)

		self.maximum_num_events_to_process = rospy.get_param(
			"~maximum_num_events_to_process", 800  # s
		)

		self.publish_jacobian_computed = rospy.get_param(
			"~publish_jacobian", True
		)

		self.publish_events_analyzed = rospy.get_param(
			"~publish_events_analyzed", False
		)

		self.is_iae_published = rospy.get_param(
			"~publish_image_acc_events", True
		)

		self.upper_limit_x = self.bounding_box_center_x + (self.bounding_box_width // 2)
		self.upper_limit_y = self.bounding_box_center_y + (self.bounding_box_height // 2)
		self.lower_limit_x = self.bounding_box_center_x - (self.bounding_box_width // 2)
		self.lower_limit_y = self.bounding_box_center_y - (self.bounding_box_height // 2)

		self.alpha_ema = rospy.get_param(
			"~alpha_ema", 1.)
		self.jacobian_computation_freq = rospy.get_param(
			"~jacobian_computation_freq", 100)
		self.centroid_computation_freq = rospy.get_param(
			"~centroid_computation_freq", 100.)
		self.command_rate = rospy.get_param(
			"~command_rate", 1000)  # Hz
		self.time_threshold_stop_commanding = rospy.get_param(
			"~time_threshold_stop_commanding_on_no_input", 1 / 20)  # s
		self.learning_rate = rospy.get_param(
			"~learning_rate", 0.004)
		self.max_commander_vel_saturation = rospy.get_param(
			"~max_commander_vel_saturation", 0.6)  # rad / s
		self.sharpness_function_type = rospy.get_param(
			"~sharpness_function_type", "image_area")  # rad / s
		self.kp_control_jacobian = rospy.get_param(
			"~kp_control_jacobian", 1.)
		self.kp_control_centroid = rospy.get_param(
			"~kp_control_centroid", 1.)
		self.camera_frame_id = ""

		self.current_centroid = np.array([0., 0.])
		self.prev_centroid = np.array([0., 0.])

		self.iwe_evaluator = ImageWarpedEventsEvaluator(self.camera_intrinsic_matrix,
														self.camera_intrinsic_matrix_inverse, self.sensor_size, 11,
														torch.Tensor([2., 2.]), param_to_eval=self.params_to_evaluate,  approximate_rmatrix=True,
														sharpness_function_type=self.sharpness_function_type, motion_model=self.motion_model)

		# self.centroid_tracker = EventsCentroidTracker(0.8, np.array([346/2, 260/2]))
		self.centroid_tracker = EventsCentroidTracker(0.6, np.array([self.camera_intrinsic_matrix[0, 2], self.camera_intrinsic_matrix[1, 2]]))
		self.event_buffering = EventsBuffering(use_event_time=True)

		# Signal processing setup
		cutoff_frequency = 10
		"""self.lowpass_filter_sos = iirfilter(2, cutoff_frequency, rs=15, btype='lowpass', ftype='cheby2', output='sos',
											fs=self.command_rate)
		self.sos_zs = np.zeros((self.lowpass_filter_sos.shape[0], 2, 2))  # Meant to store delays"""
		self.last_msg_received = None

		# Structures needed
		self.last_jacobian = torch.Tensor([0., 0., 0.])
		self.current_velocity = np.array([0., 0.])
		self.current_velocity_centroid = np.array([0., 0])

		# Ros Subscriber / timers etc
		rospy.Timer(rospy.Duration(1 / self.command_rate), self.issue_command_callback)
		rospy.Subscriber("/dvs/events_flattened", EventArrayFlattened, self.events_in_callback)
		if self.is_centroid_rbg_based: rospy.Subscriber("/detect_checker_board/pattern_centroid", Vector2Stamped, self.rgb_centroid_callback)
		rospy.Timer(rospy.Duration(1/self.jacobian_computation_freq), self.jacobian_computation)
		rospy.Timer(rospy.Duration(1./self.centroid_computation_freq), self.centroid_computation)


	def rgb_centroid_callback(self, msg):

		centroid_error = np.array([msg.vector.x, msg.vector.y])
		if self.is_centroid_rbg_based:
			self.current_velocity_centroid = np.array([0 / self.sensor_size[0], centroid_error[0] / self.sensor_size[1]])  # Divided by sensor_size


	def events_in_callback(self, msg):

		self.event_buffering.events_callback(msg)

	def centroid_computation(self, time_st):

		data = self.event_buffering.pull_data(self.sliding_window_time_centroid)
		centroid_error = self.prev_centroid
		if data is not None:
			if self.analyze_only_bounding_box:
				valid_idx = np.where(data[:, 0] > self.lower_limit_x, 1., 0.)
				valid_idx *= np.where(data[:, 0] < self.upper_limit_x, 1., 0.)
				valid_idx *= np.where(data[:, 1] < self.upper_limit_y, 1., 0.)
				valid_idx *= np.where(data[:, 1] > self.lower_limit_y, 1., 0.)

				data = data[valid_idx.astype(bool), :]

			if data.shape[0] > self.num_events_threshold:
				centroid_error = self.centroid_tracker.get_current_error(data)
				point = PointStamped()
				point.point.x = centroid_error[0]
				point.point.y = centroid_error[1]
				self.pub_centroid_error.publish(point)

		self.prev_centroid = centroid_error
		if not self.is_centroid_rbg_based:
			self.current_velocity_centroid = np.array([0/self.sensor_size[0], centroid_error[0]/self.sensor_size[1]]) #Divided by sensor_size

	def publish_img_acc_events(self, tensor):
		positive_events = torch.zeros(self.sensor_size)
		negative_events = torch.zeros(self.sensor_size)
		empty_channel = torch.zeros(self.sensor_size)

		positive_events_idx = tensor[:, 3].bool()
		positive_events.index_put_(
			(tensor[positive_events_idx, 1].long(), tensor[positive_events_idx, 0].long()),
			torch.tensor(1.), accumulate=True)
		negative_events.index_put_(
			(tensor[~ positive_events_idx, 1].long(), tensor[~ positive_events_idx, 0].long()),
			torch.tensor(1.), accumulate=True)

		rgb_image = cv2.merge((np.array(positive_events), np.array(empty_channel), np.array(negative_events)))
		rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)
		image_msg = self.bridge.cv2_to_imgmsg(rgb_image.astype(np.uint8), "bgr8")
		self.pub_image.publish(image_msg)

	def jacobian_computation(self, time_st):

		data = self.event_buffering.pull_data(self.sliding_window_time_jacob, remove_old_packets=False)
		self.last_msg_received = rospy.Time.now()
		jac = torch.Tensor([0., 0., 0.]).to(torch.float64)
		if data is not None:
			array_tensor = torch.Tensor(data).to(torch.float64)
			if self.analyze_only_bounding_box:
				valid_idx = torch.where(array_tensor[:, 0] > self.lower_limit_x, torch.Tensor([1]), torch.Tensor([0]))
				valid_idx *= torch.where(array_tensor[:, 0] < self.upper_limit_x, torch.Tensor([1]), torch.Tensor([0]))
				valid_idx *= torch.where(array_tensor[:, 1] < self.upper_limit_y, torch.Tensor([1]), torch.Tensor([0]))
				valid_idx *= torch.where(array_tensor[:, 1] > self.lower_limit_y, torch.Tensor([1]), torch.Tensor([0]))

				array_tensor = array_tensor[valid_idx.bool(), :]

			if array_tensor.shape[0] > self.num_events_threshold:
				if array_tensor.shape[0] > self.maximum_num_events_to_process: # DOWNSAMPLE IT
					array_tensor[::array_tensor.shape[0]//self.maximum_num_events_to_process]
				jac = self.iwe_evaluator.jacobian_loss_fn(torch.Tensor([0., 0., 0.]).to(torch.float64), array_tensor)

			if array_tensor.shape[0] > 0 and  self.is_iae_published:
				self.publish_img_acc_events(array_tensor)

			if self.publish_events_analyzed:
				events_array = EventArrayFlattened()
				events_array.header.stamp = self.last_msg_received
				events_array.events = array_tensor.flatten().tolist()
				self.analyzed_event_pub.publish(events_array)

		if self.motion_model == "translation":
			jac = torch.Tensor([jac[1], jac[0], jac[2]])

		jac = torch.clip(jac, -self.jacobian_clipping_value, self.jacobian_clipping_value)
		self.last_jacobian = (1 - self.alpha_ema) * self.last_jacobian + self.alpha_ema * jac
		jacobian_np_array = np.array([float(self.last_jacobian[0]), float(self.last_jacobian[1])])
		self.current_velocity = self.kp_control_jacobian * (self.current_velocity + self.learning_rate * jacobian_np_array) + self.kp_control_centroid * self.current_velocity_centroid
		if self.publish_jacobian_computed:
			point = PointStamped()
			point.point.x = self.last_jacobian[0]
			point.point.y = self.last_jacobian[1]
			self.raw_velocity_pub.publish(point)

	def issue_command_callback(self, timer_obj: rospy.timer.TimerEvent):

		if self.last_msg_received is None:
			return None

		time_now = rospy.Time().now()
		time_since_last_jac_msg = (time_now - self.last_msg_received).to_sec()
		if time_since_last_jac_msg > self.time_threshold_stop_commanding:
			return None

		# Signal processing
		# self.last_feat_error, self.sos
		wXY_wo_gain_unfiltered = self.current_velocity

		"""wXY_wo_gain_filtered, self.sos_zs = sosfilt(self.lowpass_filter_sos, wXY_wo_gain_unfiltered.reshape(-1, 1),
													axis=-1, zi=self.sos_zs)"""

		wXY_with_gain_filtered = wXY_wo_gain_unfiltered

		pub_msg = TwistStamped()
		pub_msg.header.stamp = time_now
		pub_msg.header.frame_id = self.camera_frame_id

		# Saturate velocities at the command level
		wXY_filtered_saturated = np.clip(wXY_with_gain_filtered, -self.max_commander_vel_saturation, self.max_commander_vel_saturation)

		# pub_msg.twist.angular.x = wXY_filtered_saturated[0]
		pub_msg.twist.angular.x = wXY_filtered_saturated[0]
		pub_msg.twist.angular.y = wXY_filtered_saturated[1]

		# NaNs for dimensions not commanded
		pub_msg.twist.linear.x = np.nan
		pub_msg.twist.linear.y = np.nan
		pub_msg.twist.linear.z = np.nan
		pub_msg.twist.angular.z = np.nan

		self.pub_angular_vels_.publish(pub_msg)


if __name__ == "__main__":
	rospy.init_node('gaze_lock_commander')
	CommandGazeLock()
	rospy.spin()
