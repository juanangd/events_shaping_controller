#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from events_shaping_controller_msgs.msg import Vector2Stamped, Vector2
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class ArucoMakerDetector:
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

        self.aruco_dictionary = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }

        self.image_topic = rospy.get_param('~image_topic', '/dvs/image_raw')
        self.is_image_published = rospy.get_param("~/is_image_published", True)
        self.pattern_centroid_wrt_center = rospy.get_param("~/pattern_centroid_wrt_center", True)
        self.aruco_type = rospy.get_param("~/aruco_type", "DICT_ARUCO_ORIGINAL")

        # Check that we have a valid ArUco marker
        if self.aruco_dictionary.get(self.aruco_type, None) is None:
            rospy.logerr(f"The desired aruco to detect is not existing")
            rospy.signal_shutdown("The desired aruco to detect is not existing")

        # Load the ArUco dictionary
        rospy.logerr("[INFO] detecting '{}' markers...".format(self.aruco_type))
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dictionary[self.aruco_type])
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)


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

        # Camera center
        self.camera_center = np.array([self.camera_intrinsic_matrix[0, 2], self.camera_intrinsic_matrix[1, 2]])

        try:
            sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
            self.is_algorithm_ongoing = sync_msg.data
        except rospy.ROSException:
            self.is_algorithm_ongoing = False

    def print_statistics(self, time_st):

        if self.is_algorithm_ongoing and len(self.errors)>0:
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
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = self.aruco_detector.detectMarkers(gray)
        # rospy.logerr(corners)
        # If found, draw corners
        if len(corners) > 0:
            # Flatten the ArUco IDs list
            ids = ids.flatten()

            # Loop over the detected ArUco corners
            for (marker_corner, marker_id) in zip(corners, ids):
                # Extract the marker corners
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                # Convert the (x,y) coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Draw the bounding box of the ArUco detection
                cv2.line(cv_image, top_left, top_right, (0, 255, 0), 2)
                cv2.line(cv_image, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(cv_image, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(cv_image, bottom_left, top_left, (0, 255, 0), 2)

                # Calculate and draw the center of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(cv_image, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the video frame
                # The ID is always located at the top_left of the ArUco marker
                """ cv2.putText(frame, str(marker_id),
                            (top_left[0], top_left[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)"""


            if self.is_image_published:
                output_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                self.image_pub.publish(output_image_msg)

            error_pattern = np.array([float(center_x), float(center_y)])
            if self.pattern_centroid_wrt_center:
                error_pattern -= self.camera_center

            point2d_msg = Vector2Stamped()
            point2d_msg.header = msg.header
            point2d_msg.vector.x = error_pattern[0]
            point2d_msg.vector.y = error_pattern[1]
            self.centroid_pub.publish(point2d_msg)

if __name__ == '__main__':

    rospy.init_node('checkerboard_detector', anonymous=True)
    ArucoMakerDetector()
    rospy.spin()