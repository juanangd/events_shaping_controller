#!/usr/bin/env python3

"""This node is intended to analyze the stream of events when fixation is under/over fixation"""
import rospy
import numpy as np
from scipy.signal import iirfilter, sosfilt


from events_shaping_controller_msgs.msg import Vector2
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image


from dvs_msgs.msg import EventArray, Event
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv


class CommandGazeLock():

    def __init__(self):

        self.idx = 0
        self.current_wXY = None
        self.current_vXY = None

        # Define the subscriber to events
        try:
            camera_stream = rospy.wait_for_message("/dvs/events", EventArray, timeout=5.0)
            self.sensor_size = np.array([camera_stream.height, camera_stream.width])
            rospy.logerr(f"The sensor size is received {self.sensor_size}")

        except rospy.ROSException:
            rospy.logerr("Gaze lock commander could not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("Gaze lock commander could not receive camera stream!")
            return

        self.camera_frame_id = "0"

        self.max_commander_vel_saturation = rospy.get_param(
            "~max_commander_vel_saturation", 0.2)  # rad / s

        self.current_time_surface = np.zeros(self.sensor_size)
        self.current_on_map = np.zeros(self.sensor_size)
        self.current_off_map = np.zeros(self.sensor_size)

        self.pub_on_off_events = rospy.Publisher("~on_off_events", Image, queue_size=10)
        self.pub_time_surface = rospy.Publisher("~time_surface", Image, queue_size=10)
        self.pub_angular_vels_ = rospy.Publisher("~wXY", TwistStamped, queue_size=10)


        rospy.Timer(rospy.Duration(1/20), self.publish_maps_and_reset)

        self.events_callback(camera_stream)
        self.bridge = CvBridge()

        self.command_rate = rospy.get_param(
            "~command_rate", 1000)  # Hz
        self.time_threshold_stop_commanding = rospy.get_param(
            "~time_threshold_stop_commanding_on_no_input", 1 / 5)  # s

        # Signal processing setup
        cutoff_frequency_w = 10
        self.low_pass_filter_w_sos = iirfilter(30, cutoff_frequency_w, btype='lowpass', ftype='butter', output='sos',
                                               fs=self.command_rate)
        self.sos_w_zs = np.zeros((self.low_pass_filter_w_sos.shape[0], 2, 2))  # Meant to store delays

        self.last_msg_rcvd_time = rospy.Time.now()
        rospy.Subscriber("/dvs/events", EventArray, self.events_callback)
        rospy.Subscriber('/heuristic_commander/vXY', Vector2, self.vXY_callback)


        # NOTE: This won't be real-time! This only aims to match expected rate
        self.issue_command_callback_timer_ = rospy.Timer(
            rospy.Duration(1 / self.command_rate), self.issue_command_callback)

    def publish_maps_and_reset(self, timer):

        on_off_image = np.zeros((*self.sensor_size, 3))

        on_off_image[:, :, 1] = self.current_on_map
        on_off_image[:, :, 2] = self.current_off_map

        self.current_off_map = np.zeros(self.sensor_size)
        self.current_on_map = np.zeros(self.sensor_size)

        time_surf_plot = self.current_time_surface - np.min(self.current_time_surface)
        self.current_time_surface = np.zeros(self.sensor_size)

        on_off_image = (on_off_image * 255).astype(np.uint8)
        on_off_image_msg = self.bridge.cv2_to_imgmsg(on_off_image, encoding="bgr8")
        self.pub_on_off_events.publish(on_off_image_msg)

        time_surf_norm = cv.normalize(time_surf_plot, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                      dtype=cv.CV_32F)
        time_surf_msg = self.bridge.cv2_to_imgmsg(time_surf_norm, encoding="passthrough")

        self.pub_time_surface.publish(time_surf_msg)

    @staticmethod
    def image_crop(image, center, width, height):

        left = center[0] - width // 2
        top = center[1] - height // 2
        right = left + width
        bottom = top + height

        # Crop the image
        cropped_img = image[top:bottom, left:right]
        return cropped_img

    @staticmethod
    def image_crop_bgr(image, center, width, height):

        left = center[0] - width // 2
        top = center[1] - height // 2
        right = left + width
        bottom = top + height

        # Crop the image
        cropped_img = image[top:bottom, left:right, :]
        return cropped_img

    def vXY_callback(self, data):

        self.current_vXY = np.array([data.x, data.y])
        wXY_def = np.array([-data.y, -data.x])

        # wXY_unfiltered = np.multiply(wXY_def, self.gain_rotation)
        self.gain_rotation = np.array([.6, .6])

        wXY_unfiltered = np.multiply(wXY_def, self.gain_rotation)

        wXY_filtered, self.sos_w_zs = sosfilt(self.low_pass_filter_w_sos, wXY_unfiltered.reshape(-1, 1), axis=-1,
                                              zi=self.sos_w_zs)

        self.current_wXY = wXY_filtered


    def issue_command_callback(self, timer_obj: rospy.timer.TimerEvent):

        time_now = rospy.Time().now()
        time_since_last_msg = (time_now - self.last_msg_rcvd_time).to_sec()
        self.last_msg_rcvd_time = time_now

        if time_since_last_msg > self.time_threshold_stop_commanding:
            rospy.logerr(f"fime_since_last_msg: {time_since_last_msg}")
            self.issue_commands_flag = False
            rospy.logwarn_throttle(1.0,
                                   f"Gaze lock commander stopped issuing commands as time since last input is {time_since_last_msg:.2f} s")
            return

        if self.current_wXY is None:
            return
        pub_msg = TwistStamped()
        pub_msg.header.stamp = time_now
        pub_msg.header.frame_id = self.camera_frame_id

        # Saturate velocities at the command level
        MAX_VEL_ = self.max_commander_vel_saturation  # rad / s
        wXY_filtered_saturated = np.clip(self.current_wXY, -MAX_VEL_, MAX_VEL_)
        pub_msg.twist.angular.x = wXY_filtered_saturated[0]
        pub_msg.twist.angular.y = wXY_filtered_saturated[1]

        # NaNs for dimensions not commanded
        pub_msg.twist.linear.x = np.nan
        pub_msg.twist.linear.y = np.nan
        pub_msg.twist.linear.z = np.nan
        pub_msg.twist.angular.z = np.nan

        self.pub_angular_vels_.publish(pub_msg)


    def events_callback(self, msg):

        for ev in msg.events:
            self.current_time_surface[ev.y, ev.x] = ev.ts.to_time()
            if ev.polarity:
                self.current_on_map[ev.y, ev.x] =1
            else:
                self.current_off_map[ev.y, ev.x] =1


if __name__ == "__main__":
    rospy.init_node('gaze_lock_commander')
    CommandGazeLock()
    rospy.spin()
