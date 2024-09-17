#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool


class StartSync :

    def __init__(self):

        self.pub = rospy.Publisher("sync_message", Bool, queue_size=10)
        rospy.sleep(5)

        msg = Bool()
        msg.data = True
        self.pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('start_sync')
    StartSync()
    rospy.spin()
