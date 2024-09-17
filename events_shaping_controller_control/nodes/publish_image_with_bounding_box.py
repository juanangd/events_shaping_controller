#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv



class ImagePublisherWithBoundingBox:

    def __init__(self):

        self.pub_frame = rospy.Publisher('~img_frame_init', Image, queue_size=10)

        # Define the subscriber to events
        try:
            camera_info = rospy.wait_for_message("/dvs/camera_info", CameraInfo, timeout=10.0)
            self.camera_intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
            self.camera_intrinsic_matrix_inverse = np.linalg.inv(self.camera_intrinsic_matrix)
            self.sensor_size = (camera_info.height, camera_info.width)

        except rospy.ROSException:
            rospy.logerr("ImagePublisherWithBoundingBox not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("ImagePublisherWithBoundingBox could not receive camera stream!")
            return

        self.use_camera_center_real = rospy.get_param(
            "bounding_box/bounding_box_center_is_camera_center", False)

        self.bounding_box_height = rospy.get_param(
            "bounding_box/bounding_box_height", 100
        )

        self.bounding_box_width = rospy.get_param(
            "bounding_box/bounding_box_width", 100
        )

        if self.use_camera_center_real:
            self.bounding_box_center_x = np.round(self.camera_intrinsic_matrix[0, 2]).astype(int)
            self.bounding_box_center_y = np.round(self.camera_intrinsic_matrix[1, 2]).astype(int)
        else:
            self.bounding_box_center_x = rospy.get_param(
                "bounding_box/bounding_box_center_x", self.sensor_size[1] // 2)

            self.bounding_box_center_y = rospy.get_param(
                "bounding_box/bounding_box_center_y", self.sensor_size[0] // 2)

        self.bridge = CvBridge()
        self.frame_subscriber = rospy.Subscriber("/dvs/image_raw", Image, self.frame_image_callback)

    def frame_image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Get image dimensions
        height, width, _ = cv_image.shape

        rect_x = int(self.bounding_box_center_x - self.bounding_box_width//2)
        rect_y = int(self.bounding_box_center_y - self.bounding_box_height//2)
        # Add the rectangle to the image

        cross_size = 20
        color_cross = (0, 0, 255)
        cross_thickness = 2
        cv.line(cv_image, (self.bounding_box_center_x-cross_size, self.bounding_box_center_y), (self.bounding_box_center_x+cross_size, self.bounding_box_center_y), color_cross, cross_thickness)
        cv.line(cv_image, (self.bounding_box_center_x, self.bounding_box_center_y - cross_size), (self.bounding_box_center_x, self.bounding_box_center_y + cross_size), color_cross, cross_thickness)
        cv.rectangle(cv_image, (rect_x, rect_y), (rect_x + self.bounding_box_width, rect_y + self.bounding_box_height), (0, 255, 0), 2)

        # Convert the modified image back to a ROS image message
        modified_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

        # Publish the modified image
        self.pub_frame.publish(modified_msg)

if __name__ == "__main__":

    rospy.init_node('image_publisher_with_bounding_box')
    ImagePublisherWithBoundingBox()
    rospy.spin()
