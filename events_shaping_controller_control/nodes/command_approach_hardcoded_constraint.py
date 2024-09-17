#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64, Bool
from smooth_trajectory_generator import SmoothConstantTrajectory

class CommandHeuristic():

    def __init__(self):

        # Publisher created first such that subscriber can get everything published.
        self.pub_approach_commands_ = rospy.Publisher("~vZ", TwistStamped, queue_size=10)
        self.pub_current_dist_ = rospy.Publisher("~current_distance", Float64, queue_size=10)
        self.pub_current_div_ = rospy.Publisher("~current_divergence", Float64, queue_size=10)

        try:
            sync_msg = rospy.wait_for_message("sync_message", Bool, timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("Command Heuristic Approach Constraint combined didn't receive sync message")
            rospy.signal_shutdown("Command Heuristics combined didn't receive sync message")

        rospy.loginfo(f"Command heuristics starts {rospy.Time.now()}")

        command_rate = rospy.get_param("~command_rate", 1000)  # Hz
        self.time_threshold_stop_commanding = rospy.get_param("~time_threshold_stop_commanding_on_no_input", 1 / 100)  # s

        self.smooth_trajectory_generator = SmoothConstantTrajectory(smoothing_time=0.5)

        self.camera_frame_id = "0"

        self.Z0 = 0.7
        self.initial_velocity = 0.1
        self.smoothing_time = 0.5
        self.first_time_constraint = True
        self.time_in_the_beginning = None

        self.prev_velocity = 0
        self.current_distance_to_surface = self.Z0

        # NOTE: This won't be real-time! This only aims to match expected rate
        self.issue_command_callback_timer_ = rospy.Timer(rospy.Duration(1 / command_rate), self.issue_command_callback)
        self.updating_status_timer_ = rospy.Timer(rospy.Duration(1 / 100), self.updating_status_callback)

    def updating_status_callback(self, timer_obj):
        msg_distance = Float64()
        msg_distance.data= self.current_distance_to_surface
        self.pub_current_dist_.publish(msg_distance)

        current_div = self.prev_velocity / self.current_distance_to_surface
        msg_div = Float64()
        msg_div.data = current_div
        self.pub_current_div_.publish(msg_div)

    def issue_command_callback(self, timer_obj):
        if timer_obj.last_real is not None:
            dt = (timer_obj.current_real - timer_obj.last_real).to_sec()
            distance_travelled = np.abs(dt * self.prev_velocity)
            self.current_distance_to_surface -= distance_travelled

        time_now = timer_obj.current_real
        if self.time_in_the_beginning is None:
            self.time_in_the_beginning = time_now

        time_elapsed_since_beginning = (time_now - self.time_in_the_beginning).to_sec()
        vZ_approach = np.array([0.])

        if time_elapsed_since_beginning < self.smoothing_time:
            vZ_approach[0] = self.initial_velocity * self.smooth_trajectory_generator.signal_at(time_elapsed_since_beginning)
            self.Z0_constraint = self.current_distance_to_surface
        elif self.first_time_constraint:
            self.time_in_the_beginning_constraint = time_now
            elapsed_time_constraint = (time_now - self.time_in_the_beginning_constraint).to_sec()

            self.divergence_value = self.prev_velocity / self.current_distance_to_surface
            self.first_time_constraint = False
            vZ_approach[0] = np.abs(-self.divergence_value * self.Z0_constraint * np.exp(-self.divergence_value * elapsed_time_constraint))
        else:
            elapsed_time_constraint = (time_now - self.time_in_the_beginning_constraint).to_sec()
            vZ_approach[0] = np.abs(-self.divergence_value * self.Z0_constraint * np.exp(
                -self.divergence_value * elapsed_time_constraint))

        vZ_saturated = np.clip(vZ_approach, -0.2, 0.2)
        self.prev_velocity = vZ_saturated[0]

        pub_msg = TwistStamped()
        pub_msg.header.stamp = time_now
        pub_msg.header.frame_id = self.camera_frame_id

        pub_msg.twist.linear.x = np.nan
        pub_msg.twist.linear.y = np.nan
        pub_msg.twist.linear.z = vZ_saturated[0]

        # NaNs for dimensions not commanded
        pub_msg.twist.angular.x = np.nan
        pub_msg.twist.angular.y = np.nan
        pub_msg.twist.angular.z = np.nan

        self.pub_approach_commands_.publish(pub_msg)


if __name__ == "__main__":
    rospy.init_node('approach_commander')
    CommandHeuristic()
    rospy.spin()
