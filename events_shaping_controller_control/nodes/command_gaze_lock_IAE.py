#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.signal import iirfilter, sosfilt

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image

import threading

from dvs_msgs.msg import EventArray, Event
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool, Float64
import cv2 as cv


class CommandGazeLock():

    def __init__(self):

        # Define the subscriber to events
        try:
            camera_stream = rospy.wait_for_message("/dvs/events", EventArray, timeout=5.0)
            self.sensor_size = np.array([camera_stream.height, camera_stream.width])
            rospy.logerr(f"The sensor size is received {self.sensor_size}")

        except rospy.ROSException:
            rospy.logerr("Gaze lock commander could not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("Gaze lock commander could not receive camera stream!")
            return

        try:
            sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("Command gaze locker combined didn't receive sync message")
            rospy.signal_shutdown("Command Heuristics combined didn't receive sync message")

        rospy.logerr(f"command gaze locker start at {rospy.Time.now()}")
        self.camera_frame_id = "0"
        self.buffer_events = deque()
        self.events_callback(camera_stream)
        self.bridge = CvBridge()
        self.current_wXY = None

        self.command_rate = rospy.get_param(
            "~command_rate", 1000)  # Hz
        self.time_threshold_stop_commanding = rospy.get_param(
            "~time_threshold_stop_commanding_on_no_input", 1 / 5)  # s
        init_rotation = rospy.get_param(
            "~initial_gain_command", 0.3)  # lambda_1
        self.gain_rotation = np.array([init_rotation, init_rotation])
        self.rotation_gain_max_step = rospy.get_param(
            "~rotation_gain_max_step", 0.5
        )
        self.max_commander_vel_saturation = rospy.get_param(
            "~max_commander_vel_saturation", 0.4)  # rad / s

        # Signal processing setup
        cutoff_frequency_w = 10
        self.low_pass_filter_w_sos = iirfilter(30, cutoff_frequency_w, btype='lowpass', ftype='butter', output='sos',
                                               fs=self.command_rate)
        self.sos_w_zs = np.zeros((self.low_pass_filter_w_sos.shape[0], 2, 2))  # Meant to store delays

        self.last_msg_rcvd_time = rospy.Time.now()

        self.last_x_values = deque(maxlen=3)
        self.last_y_values = deque(maxlen=3)
        self.old_gain_rotation = np.array([None, None])
        self.change_der = 0
        self.cycle_change_time_t0 = np.array([None, None])
        self.cycle_change_time_t1 = np.array([None, None])
        self.gradient_descent_previous_error = np.array([None, None])
        self.gradient_descent_max_error = np.array([0, 0])
        self.image_center = (np.flip(self.sensor_size, 0) / 2).astype(int)

        self.min_loss = np.inf
        self.best_distance_estimate = None

        self.pub_num_fired_pixels = rospy.Publisher('~num_fired_pixels', Float64, queue_size=10)
        self.pub_image_area = rospy.Publisher('~image_area', Float64, queue_size=10)
        self.pub_variance_laplacian = rospy.Publisher("~variance_laplacian", Float64, queue_size=10)
        self.pub_frame = rospy.Publisher('~img_frame_init', Image, queue_size=10)
        self.pub_loss_fn = rospy.Publisher('~IAE_alignment_metric', Float64, queue_size=10)
        self.pub_angular_vels_ = rospy.Publisher("~wXY", TwistStamped, queue_size=10)
        self.pub_ditance = rospy.Publisher('~distance_estimated', Float64, queue_size=10)
        self.pub_best_distance = rospy.Publisher("~optimal_distance", Float64, queue_size=10)
        self.pub_ground_truth = rospy.Publisher("~ground_truth_distance", Float64, queue_size=10)
        # self.groud_truth_distance = 0.489 - 0.22  # Pattern on the cylinder
        # self.groud_truth_distance = 0.489 - 0.064 # Pattern on the cube
        self.groud_truth_distance = 0.489  # Pattern on the table

        self.subscriber = rospy.Subscriber('/dvs/events', EventArray, self.events_callback)
        self.subscriber_frame = rospy.Subscriber('/dvs/image_raw', Image, self.frame_image_callback)
        self.subscriber = rospy.Subscriber('/heuristic_commander/vXYandwZ', TwistStamped, self.vXY_callback)
        self.time_window = 5.0
        self.pub_accumulated_all = rospy.Publisher('~accumulated_all', Image, queue_size=10)
        self.pub_accumulated_on_off = rospy.Publisher('~accumulated_on_off', Image, queue_size=10)


        # NOTE: This won't be real-time! This only aims to match expected rate
        self.issue_command_callback_timer_ = rospy.Timer(
            rospy.Duration(1 / self.command_rate), self.issue_command_callback)

    def callback_maximum_vxvy(self, axis):

        if self.cycle_change_time_t0[axis] is None:
            self.cycle_change_time_t0[axis] = rospy.Time.now()
            return
        else:
            self.cycle_change_time_t1[axis] = rospy.Time.now()

        rospy.logerr(f"time elapsed: {(self.cycle_change_time_t1[axis] - self.cycle_change_time_t0[axis]).to_sec()}")
        self.gradient_descent_step(self.cycle_change_time_t0[axis], self.cycle_change_time_t1[axis], axis)
        self.cycle_change_time_t0[axis] = self.cycle_change_time_t1[axis]

    @staticmethod
    def loss_fn(image_patch):
        # if image area
        F = lambda x: 1 - np.exp(-x)
        return np.sum(F(image_patch))
        # return np.count_nonzero(image_patch)

    def accumulate_events_between(self, t0, t1):
        image_acc_events_all = np.zeros(self.sensor_size)
        image_acc_events_on = np.zeros(self.sensor_size)
        image_acc_events_off = np.zeros(self.sensor_size)
        buffer_copy = deque()
        while bool(self.buffer_events) != False:
            event = self.buffer_events.pop()
            if event.ts > t1:
                buffer_copy.appendleft(event)
            elif t0 < event.ts < t1:
                image_acc_events_all[event.y, event.x] += 1
                if event.polarity == True:
                    image_acc_events_on[event.y, event.x] += 1
                else:
                    image_acc_events_off[event.y, event.x] += 1
            else:
                self.buffer_events = buffer_copy
                break
        return image_acc_events_all, image_acc_events_on, image_acc_events_off

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

    def gradient_descent_step(self, t0, t1, axis):

        image_all, image_on, image_off = self.accumulate_events_between(t0, t1)
        crop_image = self.image_crop(image_all, self.image_center, 100, 100)

        acc_norm_all = cv.normalize(crop_image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_32F)
        msg_to_publish_all = self.bridge.cv2_to_imgmsg(acc_norm_all.astype(np.uint8), encoding='mono8')
        self.pub_accumulated_all.publish(msg_to_publish_all)

        # acc_norm_on = cv.normalize(image_on, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_32F)
        # acc_norm_off = cv.normalize(image_off, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_32F)

        on_off_image = np.zeros((image_on.shape[0], image_on.shape[1], 3), dtype=np.uint8)
        on_off_image[:, :, 0] = image_off
        on_off_image[:, :, 2] = image_on

        on_off_crop = self.image_crop_bgr(on_off_image, self.image_center, 200, 200)
        acc_norm_off = cv.normalize(on_off_crop, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC3)

        msg_to_publish_on_off = self.bridge.cv2_to_imgmsg(acc_norm_off, encoding='bgr8')
        self.pub_accumulated_on_off.publish(msg_to_publish_on_off)

        current_loss = self.loss_fn(crop_image)
        if current_loss < self.min_loss:
            self.min_loss = current_loss
            self.best_distance_estimate = 1/self.gain_rotation[1]

        msg_best_d = Float64()
        msg_best_d.data = self.best_distance_estimate

        self.pub_best_distance.publish(msg_best_d)
        self.pub_loss_fn.publish(current_loss)
        self.pub_ditance.publish(1 / self.gain_rotation[1])
        self.pub_ground_truth.publish(self.groud_truth_distance)
        if self.gradient_descent_previous_error[axis] is None:
            self.gradient_descent_previous_error[axis] = current_loss
            self.old_gain_rotation[axis] = self.gain_rotation[axis]
            self.gain_rotation[axis] = self.gain_rotation[axis] + self.rotation_gain_max_step
            return
        else:
            self.gradient_descent_max_error[axis] = max(current_loss, self.gradient_descent_max_error[axis])
            """error_ratio = current_loss/self.gradient_descent_max_error
            if error_ratio < 0.5:
                error_ratio=error_ratio ** 2"""

            #error_ratio = 1 / (1 + np.exp(-12 * (current_loss / self.gradient_descent_max_error[axis] - 0.5)))
            error_ratio = current_loss / self.gradient_descent_max_error[axis]
            learning_rate = error_ratio * self.rotation_gain_max_step
            fn_derivative = np.sign(
                (current_loss - self.gradient_descent_previous_error[axis]) / (self.gain_rotation[axis] - self.old_gain_rotation[axis]))
            self.gain_rotation[axis] = self.gain_rotation[axis] - (learning_rate * fn_derivative)
            self.gain_rotation = np.where(self.gain_rotation<0, 0, self.gain_rotation)
            self.old_gain_rotation[axis] = self.gain_rotation[axis]
            self.gradient_descent_previous_error[axis] = current_loss

    def vXY_callback(self, data):

        self.last_x_values.append(data.twist.linear.x)
        self.last_y_values.append(data.twist.linear.y)

        if len(self.last_x_values) == 3:
            x_deriv = np.diff(self.last_x_values)
            if np.sign(x_deriv[0]) != np.sign(x_deriv[1]):
                rospy.logerr("sign_change x detected")
                new_thread = threading.Thread(target=self.callback_maximum_vxvy, args=(1,))
                new_thread.start()

        if len(self.last_y_values) == 3:
            y_deriv = np.diff(self.last_y_values)
            if np.sign(y_deriv[0]) != np.sign(y_deriv[1]):
                rospy.logerr("sign_change y detected")
                new_thread = threading.Thread(target=self.callback_maximum_vxvy, args=(0,))
                new_thread.start()

        """if len(self.last_x_values) == 3:
            change every cycle
            x_deriv = np.diff(self.last_x_values)
            if np.sign(x_deriv[0]) != np.sign(x_deriv[1]):
                self.change_der+=1
            if np.sign(x_deriv[0]) != np.sign(x_deriv[1]) and self.change_der%2 == 0 and self.change_der!=0:
                rospy.logerr("sign_change detected")
                new_thread = threading.Thread(target=self.callback_maximum_vx)
                new_thread.start()"""

        wXY_def = np.array([-data.twist.linear.y, -data.twist.linear.x])

        wXY_unfiltered = np.multiply(wXY_def, self.gain_rotation)

        """wXY_filtered, self.sos_w_zs = sosfilt(self.low_pass_filter_w_sos, wXY_unfiltered.reshape(-1, 1), axis=-1,
                                              zi=self.sos_w_zs)"""
        self.current_wXY = wXY_unfiltered


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

    def frame_image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Get image dimensions
        height, width, _ = cv_image.shape

        # Calculate the coordinates for the rectangle
        rect_width = 200  # Rectangle width is 25% of the image width
        rect_height = 200  # Rectangle height is 25% of the image height
        rect_x = int((width - rect_width) / 2)  # X coordinate of the rectangle's top-left corner
        rect_y = int((height - rect_height) / 2)  # Y coordinate of the rectangle's top-left corner

        # Add the rectangle to the image
        cv.rectangle(cv_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

        # Convert the modified image back to a ROS image message
        modified_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

        # Publish the modified image
        self.pub_frame.publish(modified_msg)

    def events_callback(self, msg):

        for ev in msg.events:
            self.buffer_events.append(ev)


if __name__ == "__main__":
    rospy.init_node('gaze_lock_commander')
    CommandGazeLock()
    rospy.spin()
