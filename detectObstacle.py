#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoidance:
    def __init__(self):
        rospy.init_node('obstacle_avoidance', anonymous=True)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    def scan_callback(self, data):
        # 计算检测的角度范围，左右40度
        center_index = len(data.ranges) // 2
        angle_range = 40  # 检测的角度范围（度）
        degrees_per_index = data.angle_increment * 180.0 / 3.14159  # 每个索引的角度
        index_range = int(angle_range / degrees_per_index)
        
        # 提取前方左右40度范围内的距离数据
        min_distance = float('inf')
        for i in range(center_index - index_range, center_index + index_range + 1):
            if data.ranges[i] < min_distance:
                min_distance = data.ranges[i]

        # 假设机器人正前方距离的阈值设置为0.6米
        if min_distance < 0.6:
            rospy.loginfo("Obstacle detected at 60cm within +-40 degrees! Stopping robot.")
            # 如果检测到障碍物，发送停止指令
            self.stop_robot()

    def stop_robot(self):
        stop_message = Twist()
        stop_message.linear.x = 0
        stop_message.angular.z = 0
        self.cmd_vel_pub.publish(stop_message)
        rospy.loginfo("Obstacle detected and robot stopped.")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    oa_node = ObstacleAvoidance()
    oa_node.run()
