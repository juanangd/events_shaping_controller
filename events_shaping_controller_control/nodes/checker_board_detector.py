#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from events_shaping_controller_msgs.msg import Vector2Stamped, Vector2
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class CheckerboardDetector:
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

        self.image_topic = rospy.get_param('~image_topic', '/dvs/image_raw')
        self.is_image_published = rospy.get_param("~/is_image_published", True)
        self.pattern_centroid_wrt_center = rospy.get_param("~/pattern_centroid_wrt_center", True)

        # Set up a subscriber to the image topic
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)

        # Set up a publisher for the output image
        self.image_pub = rospy.Publisher("~image_with_centroid", Image, queue_size=10)
        self.centroid_pub = rospy.Publisher("~pattern_centroid", Vector2Stamped, queue_size=10)
        self.mean_error_pub = rospy.Publisher("~mean_error", Vector2, queue_size=10)
        self.std_error_pub = rospy.Publisher("~std_error", Vector2, queue_size=10)


        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Data structures to save errors
        self.errors = []
        self.is_algorithm_ongoing = False

        # Timers
        self.timer_print_statistics = rospy.Timer(rospy.Duration(1/5), self.print_statistics)

        # Checkerboard dimensions (inner corners per a chessboard row and column)
        self.checkerboard_size = (4, 4)  # You can change this to match your checkerboard
        #self.camera_center = self.sensor_size
        self.camera_center = np.array([self.camera_intrinsic_matrix[0, 2], self.camera_intrinsic_matrix[1, 2]])

        try:
            sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
            self.is_algorithm_ongoing = sync_msg.data
        except rospy.ROSException:
            self.is_algorithm_ongoing = False

    def print_statistics(self, time_st):

        if self.is_algorithm_ongoing:
            errors_abs_val = np.abs(np.array(self.errors))
            mean_errors = np.mean(errors_abs_val, axis=0)
            std_errors = np.std(errors_abs_val, axis=0)
            mean_error_msg = Vector2()
            mean_error_msg.x = mean_errors[0]
            mean_error_msg.y = mean_errors[1]
            self.mean_error_pub.publish(mean_error_msg)

            std_error_msg =Vector2()
            std_error_msg.x = std_errors[0]
            std_error_msg.y = std_errors[1]
            self.std_error_pub.publish(std_error_msg)



    def image_callback(self, msg):

        # Convert the ROS Image message to a CV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = np.where(gray < 50, 0, np.where(gray > 200, 255, 125)).astype(np.uint8)

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

        # If found, draw corners
        if ret:
            mean_corners = np.mean(corners, axis=0).flatten()
            cv2.circle(cv_image, mean_corners.astype(int), 2, (0, 255, 0), -1)
            error_pattern = np.array(mean_corners)
            if self.pattern_centroid_wrt_center:
                error_pattern -= self.camera_center
            # cv2.drawChessboardCorners(cv_image, self.checkerboard_size, corners, ret)

            self.errors.append(error_pattern)
            if self.is_image_published:
                output_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                self.image_pub.publish(output_image_msg)

            point2d_msg = Vector2Stamped()
            point2d_msg.header = msg.header
            point2d_msg.vector.x = error_pattern[0]
            point2d_msg.vector.y = error_pattern[1]
            self.centroid_pub.publish(point2d_msg)

if __name__ == '__main__':

    rospy.init_node('checkerboard_detector', anonymous=True)
    CheckerboardDetector()
    rospy.spin()