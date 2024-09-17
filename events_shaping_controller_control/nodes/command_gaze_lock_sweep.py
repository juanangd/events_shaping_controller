#!/usr/bin/env python3

"""This node is intended to analyze the stream of events when fixation is under/over fixation"""
import matplotlib.pyplot as plt
import rospy
import numpy as np
from scipy.signal import iirfilter, sosfilt
from collections import deque
import threading

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CameraInfo, Image

from std_msgs.msg       import Bool, Float64
from events_shaping_controller_msgs.msg import ImgAccumulatedEventsDistance
from dvs_msgs.msg import EventArray, Event
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
from threading import Lock
from image_warped_events.sharpness_loss_fns import SharpnessLossFunctionSuite
import torch



class CommandGazeLock():

    def __init__(self):

        self.idx = 0
        self.current_wXY = None
        self.current_vXY = None
        self.buffer_events = deque()
        self.distance_to_object = rospy.get_param("~distance_to_object", 0.44)
        self.distance_to_object_step = rospy.get_param("~distance_to_object_step", -0.01)
        self.sharpness_fun_type = rospy.get_param("~sharpness_fun_type", "variance")
        self.measure_sharpness_bounding_box = rospy.get_param("~measure_sharpness_bounding_box", True)

        self.bounding_box_height = rospy.get_param(
            "bounding_box/bounding_box_height", 100
        )

        self.bounding_box_width = rospy.get_param(
            "bounding_box/bounding_box_width", 250
        )

        self.bounding_box_center_x = rospy.get_param(
            "bounding_box/bounding_box_center_x", 173
        )

        self.bounding_box_center_y = rospy.get_param(
            "bounding_box/bounding_box_center_y", 130
        )

        self.upper_limit_x = self.bounding_box_center_x + (self.bounding_box_width // 2)
        self.upper_limit_y = self.bounding_box_center_y + (self.bounding_box_height // 2)
        self.lower_limit_x = self.bounding_box_center_x - (self.bounding_box_width // 2)
        self.lower_limit_y = self.bounding_box_center_y - (self.bounding_box_height // 2)

        self.sharpness_fun = SharpnessLossFunctionSuite(self.sharpness_fun_type).fn_loss

        self.idx = 0

        self.locker = Lock()

        self.cycle_change_time_t0 = np.array([None, None])
        self.cycle_change_time_t1 = np.array([None, None])

        self.last_x_values = deque(maxlen=3)
        self.last_y_values = deque(maxlen=3)


        self.pub_angular_vels_ = rospy.Publisher("~wXY", TwistStamped, queue_size=10)
        self.pub_iae_distance = rospy.Publisher('~image_accumulated_events_distance', ImgAccumulatedEventsDistance, queue_size=10)
        self.pub_sharpness_function = rospy.Publisher("~sharpness_measure", Float64, queue_size=10)
        self.pub_image_acc_events = rospy.Publisher("~image_accumulaed_events", Image, queue_size=10)

        # Define the subscriber to events
        try:
            camera_stream = rospy.wait_for_message("/dvs/events", EventArray, timeout=5.0)
            self.sensor_size = np.array([camera_stream.height, camera_stream.width])
            rospy.loginfo(f"The sensor size is received {self.sensor_size}")

        except rospy.ROSException:
            rospy.logerr("Gaze lock commander could not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("Gaze lock commander could not receive camera stream!")
            return

        try:
            sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("CommandGazeLock didn't receive sync message")
            rospy.signal_shutdown("CommandGazeLock didn't receive sync message")

        rospy.loginfo(f"command gaze locker starts {rospy.Time.now()}")

        self.camera_frame_id = "0"

        self.max_commander_vel_saturation = rospy.get_param(
            "~max_commander_vel_saturation", 0.5)  # rad / s

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
        rospy.Subscriber('/heuristic_commander/vXYandwZ', TwistStamped, self.vXY_callback)

        # NOTE: This won't be real-time! This only aims to match expected rate
        self.issue_command_callback_timer_ = rospy.Timer(
            rospy.Duration(1 / self.command_rate), self.issue_command_callback)


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

        x_velocity = data.twist.linear.x
        y_velocity = data.twist.linear.y

        self.last_x_values.append(x_velocity)
        self.last_y_values.append(y_velocity)

        if len(self.last_x_values) == 3:
            x_deriv = np.diff(self.last_x_values)
            if np.sign(x_deriv[0]) != np.sign(x_deriv[1]):
                rospy.loginfo("sign_change x detected")
                new_thread = threading.Thread(target=self.callback_maximum_vxvy, args=(1, self.distance_to_object,))
                new_thread.start()
                self.distance_to_object += self.distance_to_object_step
                if self.distance_to_object <= 0.14:
                    rospy.signal_shutdown("Experiment is finished!!")

        if len(self.last_y_values) == 3:
            y_deriv = np.diff(self.last_y_values)
            if np.sign(y_deriv[0]) != np.sign(y_deriv[1]):
                rospy.loginfo("sign_change y detected")
                new_thread = threading.Thread(target=self.callback_maximum_vxvy, args=(0, self.distance_to_object, ))
                new_thread.start()


        wXY_def = np.array([-y_velocity, -x_velocity])
        wXY_unfiltered = np.multiply(wXY_def, 1/self.distance_to_object)
        self.current_wXY = wXY_unfiltered


    def callback_maximum_vxvy(self, axis, distance_to_object):

        if self.cycle_change_time_t0[axis] is None:
            self.cycle_change_time_t0[axis] = rospy.Time.now()
            return
        else:
            self.cycle_change_time_t1[axis] = rospy.Time.now()

        rospy.loginfo(f"time elapsed: {(self.cycle_change_time_t1[axis] - self.cycle_change_time_t0[axis]).to_sec()}")
        self.compute_publish_IAE(self.cycle_change_time_t0[axis], self.cycle_change_time_t1[axis], distance_to_object)
        self.cycle_change_time_t0[axis] = self.cycle_change_time_t1[axis]
    
        
    def compute_publish_IAE(self, t0, t1, distance_object):

        image_accumulated_event = np.zeros(self.sensor_size)

        self.locker.acquire()
        buffer_to_iterate = self.buffer_events
        self.locker.release()

        buffer_copy = deque()
        while bool(buffer_to_iterate) != False:
            event = buffer_to_iterate.pop()
            if event.ts > t1:
                buffer_copy.appendleft(event)
            elif t0 < event.ts < t1:
                image_accumulated_event[event.y, event.x] += 1
            else:
                self.locker.acquire()
                self.buffer_events = buffer_copy
                self.locker.release()
                break

        acc_norm_all = cv.normalize(image_accumulated_event, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_32F)
        iae_msg = self.bridge.cv2_to_imgmsg(acc_norm_all.astype(np.uint8), encoding='mono8')

        iae_and_distance_structure_msg = ImgAccumulatedEventsDistance()
        iae_and_distance_structure_msg.distance_to_object = distance_object
        iae_and_distance_structure_msg.image_accumulated_events = image_accumulated_event.reshape(-1)

        msg_sharpness = Float64()
        image_tensor = torch.Tensor(image_accumulated_event)
        if self.measure_sharpness_bounding_box:
            image_tensor = image_tensor[self.lower_limit_y:self.upper_limit_y, self.lower_limit_x:self.upper_limit_x]
        msg_sharpness.data = - self.sharpness_fun(image_tensor)
        self.pub_sharpness_function.publish(msg_sharpness)
        self.pub_iae_distance.publish(iae_and_distance_structure_msg)
        self.pub_image_acc_events.publish(iae_msg)

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
            self.buffer_events.append(ev)

if __name__ == "__main__":
    rospy.init_node('gaze_lock_commander')
    CommandGazeLock()
    rospy.spin()
