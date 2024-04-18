#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, PoseStamped

class Navigation:
    def __init__(self):
        rospy.init_node('navigation_node', anonymous=True)
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.current_goal = None
        self.at_goal = False
    
    def goal_callback(self, msg):
        self.current_goal = msg
        self.at_goal = False
        rospy.loginfo("New goal received: x={}, y={}".format(msg.pose.position.x, msg.pose.position.y))
        self.navigate_to_goal()
    
    def navigate_to_goal(self):
        if self.current_goal and not self.at_goal:
            move_cmd = Twist()
            sign = int(self.current_goal.pose.orientation.w)  # Assuming the sign result is stored in orientation.w for simplicity
            
            if sign == 0:
                # Random walk or rotate to search for signs
                move_cmd.linear.x = 0.1
                move_cmd.angular.z = 0.3
            elif sign == 1:
                # Turn left
                move_cmd.angular.z = 0.5
            elif sign == 2:
                # Turn right
                move_cmd.angular.z = -0.5
            elif sign == 3:
                # Turn around
                move_cmd.angular.z = 1.0
            elif sign == 4:
                # Do not enter - stop and search for new path
                move_cmd.linear.x = 0
                move_cmd.angular.z = 1.0  # Consider turning around
            elif sign == 5:
                # Goal reached
                rospy.loginfo("Goal reached!")
                move_cmd.linear.x = 0
                move_cmd.angular.z = 0
                self.at_goal = True
                rospy.signal_shutdown("Goal Reached")
            
            self.cmd_vel_pub.publish(move_cmd)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    nav_node = Navigation()
    nav_node.run()
