#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Float64
from dvs_msgs.msg import EventArray, Event, EventArrayFlattened
import torch
from image_warped_events.cmax_optimizer import CmaxOptimizer
from sensor_msgs.msg import CameraInfo

class DivergenceOptimizer:

    def __init__(self):

        try:
            camera_info = rospy.wait_for_message("/dvs/camera_info", CameraInfo, timeout=10.0)
            self.camera_intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
            self.camera_intrinsic_matrix_inverse = np.linalg.inv(self.camera_intrinsic_matrix)
            self.sensor_size = (camera_info.height, camera_info.width)

        except rospy.ROSException:
            rospy.logerr("Divergence not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("Divergence could not receive camera stream!")
            return

        self.camera_intrinsic_matrix = torch.Tensor(self.camera_intrinsic_matrix)
        self.camera_intrinsic_matrix_inverse = torch.Tensor(self.camera_intrinsic_matrix_inverse)
        self.sensor_size = torch.Size(self.sensor_size)

        self.sharpness_function_type = rospy.get_param(
            "~sharpness_fun_type", "variance"
        )

        self.motion_model = rospy.get_param(
            "~motion_model", "translation_divergence"
        )

        self.cmax_optimizer = CmaxOptimizer(self.camera_intrinsic_matrix,
                                       self.camera_intrinsic_matrix_inverse,
                                       self.sensor_size,
                                       11,
                                       torch.Tensor([2., 2.]),
                                       self.motion_model,
                                       lr=1.,
                                       stopping_criteria_tol=1e-5)


        rospy.Subscriber("/approach_commander/analyzed_events", EventArrayFlattened, self.events_packet_in_callback)
        self.pub_divergence = rospy.Publisher("~divergence", Float64, queue_size=10)

    def events_packet_in_callback(self, msg):

        events_tensor = torch.Tensor(np.array(msg.events).reshape((-1, 4))).to(torch.float64)
        best_value, losses = self.cmax_optimizer.optimize(events_tensor, torch.tensor(0.))
        msg_div = Float64()
        msg_div.data = best_value
        self.pub_divergence.publish(msg_div)


if __name__ == "__main__":

    rospy.init_node('divergence_optimizer')

    DivergenceOptimizer()
    rospy.spin()
