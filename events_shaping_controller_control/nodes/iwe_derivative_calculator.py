#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from dvs_msgs.msg import EventArray, Event, EventArrayFlattened
import torch
from events_buffering import EventsBuffering
from sensor_msgs.msg import CameraInfo
from image_warped_events.image_warped_events_evaluator import ImageWarpedEventsEvaluator

class IWEDerivator:

    def __init__(self):

        try:
            camera_info = rospy.wait_for_message("/dvs/camera_info", CameraInfo, timeout=10.0)
            self.camera_intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
            self.camera_intrinsic_matrix_inverse = np.linalg.inv(self.camera_intrinsic_matrix)
            self.sensor_size = (camera_info.height, camera_info.width)

        except rospy.ROSException:
            rospy.logerr("IWEDerivatorcould not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("IWEDerivator could not receive camera stream!")
            return

        self.camera_intrinsic_matrix = torch.Tensor(self.camera_intrinsic_matrix)
        self.camera_intrinsic_matrix_inverse = torch.Tensor(self.camera_intrinsic_matrix_inverse)
        self.sensor_size = torch.Size(self.sensor_size)

        self.bounding_box_width = rospy.get_param(
            "bounding_box/bounding_box_width", 200
        )

        self.bounding_box_height = rospy.get_param(
            "bounding_box/bounding_box_height", 100
        )

        self.bounding_box_center_x = rospy.get_param(
            "bounding_box/bounding_box_center_x", self.camera_intrinsic_matrix[0, 2]
        )

        self.bounding_box_center_y = rospy.get_param(
            "bounding_box/bounding_box_center_y", self.camera_intrinsic_matrix[0, 2]
        )

        self.alpha_ema = rospy.get_param(
            "~alpha_ema", 1.
        )

        self.event_num_threshold = rospy.get_param(
            "~event_num_threshold", 200
        )

        self.computation_loop = rospy.get_param(
            "~computation_loop_freq", 100
        )
        self.window_time = rospy.get_param(
            "~window_time", 0.02
        )

        self.sharpness_function_type = rospy.get_param(
            "~sharpness_fun_type", "variance"
        )

        self.motion_model = rospy.get_param(
            "~motion_model", "rotation"
        )

        self.param_to_eval = rospy.get_param(
            "~param_to_eval", [False, False, True]
        )

        self.maximum_num_events_to_process = rospy.get_param(
            "~maximum_num_events_to_process", 700
        )

        self.analyze_only_bounding_box = rospy.get_param(
            "~analyze_only_bounding_box", True
        )

        self.last_jacobian = torch.Tensor([0., 0., 0.])

        self.upper_limit_x = self.bounding_box_center_x + (self.bounding_box_width // 2)
        self.upper_limit_y = self.bounding_box_center_y + (self.bounding_box_height // 2)
        self.lower_limit_x = self.bounding_box_center_x - (self.bounding_box_width // 2)
        self.lower_limit_y = self.bounding_box_center_y - (self.bounding_box_height //2)

        self.event_buffering = EventsBuffering(use_event_time=True)
        self.iwe_evaluator = ImageWarpedEventsEvaluator(self.camera_intrinsic_matrix,
                                                        self.camera_intrinsic_matrix_inverse, self.sensor_size, 11,
                                                        torch.Tensor([2., 2.]), sharpness_function_type=self.sharpness_function_type,
                                                        motion_model=self.motion_model,
                                                        param_to_eval=self.param_to_eval,
                                                        approximate_rmatrix=True)

        rospy.Subscriber("/dvs/events_flattened", EventArrayFlattened, self.events_in_callback)
        rospy.Timer(rospy.Duration(1/self.computation_loop), self.compute_jacobian)
        self.pub_jac = rospy.Publisher("~jacobian", PointStamped, queue_size=10)

    def events_in_callback(self, msg):

        self.event_buffering.events_callback(msg)

    def compute_jacobian(self, timer):

        data = self.event_buffering.pull_data(self.window_time)
        if data is not None:
            array_tensor = torch.Tensor(data)
            if self.analyze_only_bounding_box:
                valid_idx = torch.where(array_tensor[:, 0] > self.lower_limit_x, torch.Tensor([1]), torch.Tensor([0]))
                valid_idx *= torch.where(array_tensor[:, 0] < self.upper_limit_x, torch.Tensor([1]), torch.Tensor([0]))
                valid_idx *= torch.where(array_tensor[:, 1] < self.upper_limit_y, torch.Tensor([1]), torch.Tensor([0]))
                valid_idx *= torch.where(array_tensor[:, 1] > self.lower_limit_y, torch.Tensor([1]), torch.Tensor([0]))

                array_tensor = array_tensor[valid_idx.bool(), :]

            if array_tensor.shape[0] > self.event_num_threshold:
                time_now = rospy.Time.now()
                if array_tensor.shape[0] > self.maximum_num_events_to_process:  # TODO: ADD PARAMS AND MAKE IT MORE USABLE!
                    randomize_idx = np.random.choice(np.arange(array_tensor.shape[0]),
                                                     self.maximum_num_events_to_process, replace=False)
                    array_tensor = array_tensor[randomize_idx, :]
                jac = self.iwe_evaluator.jacobian_loss_fn(torch.Tensor([0., 0., 0.]), array_tensor)
                jac_smoothed = (1 - self.alpha_ema) * self.last_jacobian + self.alpha_ema * jac
                self.last_jacobian = jac_smoothed
                point2pub = PointStamped()
                point2pub.header.stamp = time_now
                point2pub.point.x = jac_smoothed[0]
                point2pub.point.y = jac_smoothed[1]
                point2pub.point.z = jac_smoothed[2]
                self.pub_jac.publish(point2pub)


if __name__ == "__main__":

    rospy.init_node('derivative_evaluator')

    IWEDerivator()
    rospy.spin()
