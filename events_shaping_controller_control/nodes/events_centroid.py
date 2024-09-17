#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rospy
from dvs_msgs.msg import EventArray, Event
from geometry_msgs.msg import PointStamped

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class EventsCentroid:

    def __init__(self):

        try:
            camera_info = rospy.wait_for_message("/dvs/camera_info", CameraInfo, timeout=10.0)
            self.camera_intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
            self.camera_intrinsic_matrix_inverse = np.linalg.inv(self.camera_intrinsic_matrix)
            self.sensor_size = (camera_info.height, camera_info.width)
            rospy.logerr(f"The sensor size is received {self.sensor_size}")

        except rospy.ROSException:
            rospy.logerr("Events Counting could not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("RotationOptimizationIWE could not receive camera stream!")
            return

        self.bridge = CvBridge()


        self.loop_frequency = rospy.get_param(
            "~loop_frequency", 200
        )

        self.image_publisher_freq = rospy.get_param(
            "~image_publisher_freq", 50
        )

        self.bounding_box_width = rospy.get_param(
            "~bounding_box_width", 100
        )

        self.bounding_box_height = rospy.get_param(
            "~bounding_box_height", 100
        )

        self.bounding_box_center_x = rospy.get_param(
            "~bounding_box_center_x", self.sensor_size[1] // 2
        )

        self.bounding_box_center_y = rospy.get_param(
            "~bounding_box_center_y", self.sensor_size[0] // 2
        )

        self.number_events_threshold = rospy.get_param(
            "~number_events_threshold", 60
        )

        self.alpha_ema = rospy.get_param(
            "~alpha_ema", 0.9
        )

        self.compensation_with_angular_vel = rospy.get_param(
            "~compensation_with_angular_vel", False
        )

        self.callback = False

        self.current_on_map = np.zeros(self.sensor_size)
        self.current_off_map = np.zeros(self.sensor_size)
        self.number_events_step = 0
        self.prev_error = np.zeros((2, 1))
        self.mean_centroids_img = None


        self.upper_limit_x = self.bounding_box_center_x + (self.bounding_box_width // 2)
        self.upper_limit_y = self.bounding_box_center_y + (self.bounding_box_height // 2)
        self.lower_limit_x = self.bounding_box_center_x - (self.bounding_box_width // 2)
        self.lower_limit_y = self.bounding_box_center_y - (self.bounding_box_height //2)

        self.pub_error_centroid = rospy.Publisher("~events_centroid", PointStamped, queue_size=10)
        self.pub_image = rospy.Publisher("~events_and_centroid", Image, queue_size=10)

        rospy.Subscriber("/dvs/events", EventArray, self.events_in_callback)
        rospy.Timer(rospy.Duration(1/self.loop_frequency), self.update_measurement)
        rospy.Timer(rospy.Duration(1/self.image_publisher_freq), self.publish_image)


    @staticmethod
    def feat_locs_to_jacobian_(feat_locs):
        jacobian = np.empty((2, 2))

        x_ = feat_locs[0, 0]
        y_ = feat_locs[1, 0]

        jacobian[0, 0] = x_*y_
        jacobian[0, 1] = - (1 + x_**2)
        jacobian[1, 0] = - (1 + y_**2)
        jacobian[1, 1] = - x_ * y_

        return jacobian


    def events_in_callback(self, msg):

        for ev in msg.events:

            if self.callback:
                #thread = threading.Thread(target=self.publish_info, args=(self.current_on_map, self.current_off_map, self.current_time_surface, ))
                #thread.start()
                self.publish_info(self.current_on_map, self.current_off_map, self.number_events_step)
                self.current_on_map = np.zeros(self.sensor_size)
                self.current_off_map = np.zeros(self.sensor_size)
                self.number_events_step = 0
                self.callback = False

            if self.lower_limit_x < ev.x < self.upper_limit_x and self.lower_limit_y < ev.y < self.upper_limit_y:
                if ev.polarity:
                    self.current_on_map[ev.y, ev.x] = 1
                else:
                    self.current_off_map[ev.y, ev.x] = 1

                self.number_events_step+=1

    def publish_info(self, on_map, off_map, number_events):

        full_activation_map = on_map + off_map
        full_activation_map_filt = full_activation_map
        #full_activation_map_filt = signal.medfilt2d(full_activation_map, kernel_size=3)
        idx_activation = np.argwhere(full_activation_map_filt>0)

        self.mean_centroids_img = None
        error_centroid = PointStamped()

        if idx_activation.shape[0] != 0 and number_events > self.number_events_threshold:
            mean_centroid = np.mean(idx_activation, axis=0)
            self.mean_centroids_img = np.flip(mean_centroid)

            feat_x = self.mean_centroids_img[0] - self.sensor_size[1]//2
            feat_y = self.mean_centroids_img[1] - self.sensor_size[0]//2

            feats = np.array([feat_x, feat_y]).reshape((-1, 1))

            if self.compensation_with_angular_vel:
                feat_jacob = self.feat_locs_to_jacobian_(feats)
                iteration_matrix = np.linalg.inv(feat_jacob)
            else:
                iteration_matrix = np.eye(2)

            feat_corrected = iteration_matrix @ feats
            new_feat_corrected = self.alpha_ema * feat_corrected + (1-self.alpha_ema) * self.prev_error
            error_centroid.point.x = new_feat_corrected[0, 0]
            error_centroid.point.y = new_feat_corrected[1, 0]

            self.prev_error = new_feat_corrected
            error_centroid.header.stamp = rospy.Time.now()
            self.pub_error_centroid.publish(error_centroid)

        """else:

            error_centroid.point.x = self.alpha_ema * self.prev_error[0, 0]
            error_centroid.point.y = self.alpha_ema * self.prev_error[1, 0]"""



    def publish_image(self, timer):

        full_activation_map = self.current_on_map + self.current_off_map

        full_activation_map_norm = np.where(full_activation_map>0, 1, 0)

        color_image = cv.cvtColor((full_activation_map_norm * 255).astype(np.uint8), cv.COLOR_GRAY2RGB)
        if self.mean_centroids_img is not None:
            cv.circle(color_image, (int(self.mean_centroids_img[0]), int(self.mean_centroids_img[1])), 2, (0, 255, 0), -1)

        image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="passthrough")
        self.pub_image.publish(image_msg)


    def update_measurement(self, timer):

        self.callback = True


if __name__ == "__main__":
    rospy.init_node('events_centroid')

    EventsCentroid()
    rospy.spin()
