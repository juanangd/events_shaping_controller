#!/usr/bin/env python3
import cv2
import rospy
from dvs_msgs.msg import EventArray, Event
from events_shaping_controller_msgs.msg import Vector2
from std_msgs.msg import Header
from events_shaping_controller_msgs.msg import OpticalFlowEstimation

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2 as cv


class EventBasedOpticalFlow:
    def __init__(self):

        # Define the subscriber to events
        try:
            camera_stream = rospy.wait_for_message("/dvs/events", EventArray, timeout=5.0)
            self.sensor_size = np.array([camera_stream.height, camera_stream.width])
            self.current_time_surface = np.zeros(self.sensor_size)
            self.current_t_ref = camera_stream.events[0].ts.to_time()
            self.latest_event_timestamp = self.current_t_ref
            self.events_callback(camera_stream)
            rospy.logerr(f"The sensor size is received {self.sensor_size}")

        except rospy.ROSException:
            rospy.logerr("Gaze lock commander could not receive event-based camera string! Exiting.")
            rospy.signal_shutdown("Gaze lock commander could not receive camera stream!")
            return

        self.subscriber = rospy.Subscriber('/dvs/events', EventArray, self.events_callback)

        # Parameters
        # TODO: AVOID HARD-CODING
        self.period_loop_opt_flow_decaying = 1/50
        self.period_loop_opt_flow_relative = 1/100
        self.filter_older_events_time = 0.2 # when 0 no filter at all // inf value when time surface refresh
        self.is_gradient_published = False
        self.is_time_surface_published = False
        self.use_sobel_filter = True
        self.time_surface_decaying = False
        self.exponential_decaying = False
        self.square_size = 100

        # Define the publisher flow image
        self.publisher_coloured = rospy.Publisher('/event_based_optical_flow/coloured_flow', Image, queue_size=1)
        self.publisher_time_surface = rospy.Publisher('/event_based_optical_flow/time_surface', Image, queue_size=1)
        self.publisher_grad_x = rospy.Publisher('/event_based_optical_flow/grad_x', Image, queue_size=1)
        self.publisher_grad_y = rospy.Publisher('/event_based_optical_flow/grad_y', Image, queue_size=1)
        self.publisher_magnitudes = rospy.Publisher('/event_based_optical_flow/magnitudes', Vector2, queue_size=1)
        self.publisher_optic_flow_maps = rospy.Publisher('/event_based_optical_flow/optic_flow_stream', OpticalFlowEstimation, queue_size=1)

        # Define the time surface variables
        self.tau_decay = 0.1

        # Define bridge to convert cv images to ros msgs
        self.bridge = CvBridge()

        # Define x,y gradients kernels
        # 5-tap fixed kernel
        self.prefilt_kernel = np.array([3.342604e-2, 0.241125, 0.450898, 0.241125, 3.342604e-2]).reshape((1, -1))
        self.diff1_kernel = np.array([-9.186104e-2, -0.307610, 0., 0.307610, 9.186104e-2]).reshape((1, -1))
        self.diff2_kernel = np.array([0.202183, 9.181186e-2, -0.587989, 9.181186e-2, 0.202183]).reshape((1, -1))

        # Define timer
        if self.time_surface_decaying:
            loop_time = self.period_loop_opt_flow_decaying
        else:
            loop_time= self.period_loop_opt_flow_relative

        rospy.Timer(rospy.Duration(loop_time), self.optical_flow)

        # Area of operation variables
        self.center_x, self.center_y = int(self.sensor_size[1] / 2), int(self.sensor_size[0] / 2)
        self.square_top_left = (self.center_x - self.square_size // 2, self.center_y - self.square_size // 2)
        self.square_bottom_right = (self.center_x + self.square_size // 2, self.center_y + self.square_size // 2)

    def compute_spatial_gradients(self, image):

        if self.use_sobel_filter:
            grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
            grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
        else:

            grad_x = cv2.filter2D(image, -1, self.prefilt_kernel, borderType=cv2.BORDER_REFLECT)
            grad_x = cv2.filter2D(grad_x, -1, self.diff1_kernel, borderType=cv2.BORDER_REFLECT)
            grad_y = cv2.filter2D(image, -1, self.prefilt_kernel.reshape((-1, 1)), borderType=cv2.BORDER_REFLECT)
            grad_y = cv2.filter2D(grad_y, -1, self.diff1_kernel.reshape((-1, 1)), borderType=cv2.BORDER_REFLECT)

        if self.is_gradient_published:
            grad_x_norm = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            grad_y_norm = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            msg_to_publish_x = self.bridge.cv2_to_imgmsg(grad_x_norm.astype(np.uint8), encoding='mono8')
            msg_to_publish_y = self.bridge.cv2_to_imgmsg(grad_y_norm.astype(np.uint8), encoding='mono8')

            self.publisher_grad_x.publish(msg_to_publish_x)
            self.publisher_grad_y.publish(msg_to_publish_y)

        return grad_x, grad_y

    def optical_flow_time_surface(self, time_surface):

        grad_x, grad_y = self.compute_spatial_gradients(time_surface)

        ones = np.ones(self.sensor_size)
        vx = np.divide(ones, grad_x, out=np.zeros(self.sensor_size), where=grad_x != 0)
        vy = np.divide(ones, grad_y, out=np.zeros(self.sensor_size), where=grad_y != 0)

        vx_not_zero = vx[vx!=0]
        vy_not_zero = vy[vy!=0]
        if len(vx_not_zero)!=0:
            q75_x, q25_x = np.percentile(vx_not_zero, [85, 15])
            vx[(vx < q25_x) | (vx > q75_x)] = 0

        if len(vy_not_zero)!=0:
            q75_y, q25_y = np.percentile(vy[vy!=0], [85, 15])
            vy[(vy < q25_y) | (vy > q75_y)] = 0

        vx_filtered = cv.medianBlur(vx.astype("float32"), 5)
        vy_filtered = cv.medianBlur(vy.astype("float32"), 5)

        return vx_filtered, vy_filtered

    def optical_flow(self, timer):

        # TODO : Different implementations of optical flow
        if self.current_time_surface is not None and self.current_t_ref is not None:
            if self.time_surface_decaying:
                self.current_t_ref = self.latest_event_timestamp
                self.current_time_surface[self.current_t_ref-self.current_time_surface>self.filter_older_events_time] = 0
                if self.exponential_decaying:
                    time_surface = np.where(
                        self.current_time_surface != 0,
                        np.exp((self.current_time_surface - self.current_t_ref) / self.tau_decay),
                        0
                    )
                else:

                    time_surface = np.where(
                        self.current_time_surface != 0,
                        1 + (self.current_time_surface - self.current_t_ref) / (2 * self.tau_decay),
                        0
                    )
            else:
                time_surface = np.where(self.current_time_surface != 0,
                                                 self.current_time_surface - self.current_t_ref, 0)
                self.current_time_surface[
                    self.latest_event_timestamp - self.current_time_surface > self.filter_older_events_time] = 0
                self.current_t_ref = self.latest_event_timestamp

            if self.is_time_surface_published:
                time_surface_decay_norm = cv2.normalize(time_surface, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                msg_to_publish_time_surface = self.bridge.cv2_to_imgmsg(time_surface_decay_norm.astype(np.uint8), encoding='mono8')
                self.publisher_time_surface.publish(msg_to_publish_time_surface)


            vx_filtered, vy_filtered = self.optical_flow_time_surface(time_surface)
            vx_filtered_crop = vx_filtered[self.center_y-50:self.center_y+50, self.center_x-50:self.center_x+50]
            vy_filtered_crop = vy_filtered[self.center_y-50:self.center_y+50, self.center_x-50:self.center_x+50]

            if np.count_nonzero(vx_filtered_crop)>0 and np.count_nonzero(vy_filtered_crop)>0:
                vx_median = np.median(vx_filtered_crop[vx_filtered_crop!=0])
                vy_median = np.median(vy_filtered_crop[vy_filtered_crop!=0])
            else:
                vx_median = 0
                vy_median = 0

            coloured_optical_flow = np.zeros((*self.sensor_size, 3))
            coloured_optical_flow[..., 1] = 255
            mag, ang = cv.cartToPolar(vx_filtered, vy_filtered)

            coloured_optical_flow[:, :, 0] = ang * 180 / np.pi / 2
            coloured_optical_flow[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            self.flow_estimation_publisher(coloured_optical_flow, vx_median, vy_median, vx_filtered, vy_filtered)

    def flow_estimation_publisher(self, coloured_optical_flow, vx_median, vy_median, vx_filtered, vy_filtered):


        image_to_publish = cv.cvtColor(coloured_optical_flow.astype(np.uint8), cv.COLOR_HSV2BGR)
        cv2.rectangle(image_to_publish, self.square_top_left, self.square_bottom_right , [0,0,255], thickness=1)
        msg_to_publish_flow = self.bridge.cv2_to_imgmsg(image_to_publish, encoding='bgr8')
        msg_to_publish_vx = self.bridge.cv2_to_imgmsg(vx_filtered)
        msg_to_publish_vy  = self.bridge.cv2_to_imgmsg(vy_filtered)
        self.publisher_coloured.publish(msg_to_publish_flow)

        magnitudes_msg = Vector2()
        magnitudes_msg.x = float(vx_median)
        magnitudes_msg.y = float(vy_median)

        opt_flow_msg = OpticalFlowEstimation()
        opt_flow_msg.header = Header()
        opt_flow_msg.header.stamp = rospy.Time.now()
        opt_flow_msg.vx_map = msg_to_publish_vx
        opt_flow_msg.vy_map = msg_to_publish_vy
        opt_flow_msg.magnitudes = magnitudes_msg
        self.publisher_optic_flow_maps.publish(opt_flow_msg)


    def events_callback(self, msg):

        for ev in msg.events:
            current_time = ev.ts.to_time()
            if self.latest_event_timestamp > current_time: # Only happens when playing rosbag in loop
                self.current_time_surface = np.zeros(self.sensor_size)
                self.current_t_ref = current_time

            self.current_time_surface[ev.y, ev.x] = current_time
            self.latest_event_timestamp = current_time


if __name__ == "__main__":
    rospy.init_node('event_based_optical_flow')
    print("node")

    EventBasedOpticalFlow()
    rospy.spin()
