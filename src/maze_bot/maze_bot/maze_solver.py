import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from math import pi,cos,sin
from math import pow , atan2,sqrt , degrees,asin

from .my_path_planning import Dijkstra
from .my_localization import Map

from nav_msgs.msg import Odometry
from std_msgs.msg import String

import numpy as np
from numpy import interp



class maze_solver(Node):

    def __init__(self):
        
        super().__init__("maze_solving_node")
        
        self.velocity_publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.videofeed_subscriber = self.create_subscription(Image,'/upper_camera/image_raw',self.get_video_feed_cb,10)
        self.videofeed_subscriber_kitchen = self.create_subscription(Image,'/upper_camera_kitchen/image_raw',self.get_video_feed_cb_kitchen,10)
        self.videofeed_subscriber_bedroom = self.create_subscription(Image,'/upper_camera_bedroom/image_raw',self.get_video_feed_cb_bedroom,10)
        
        # Visualizing what the robot sees by subscribing to bot_camera/Image_raw
        self.bot_subscriber = self.create_subscription(Image,'/depth_camera/depth/image_raw',self.process_data_depth,10)
        self.bot_view_subcriber = self.create_subscription(Image,'/depth_camera/image_raw',self.process_data_raw,10)
        self.timer = self.create_timer(0.2, self.maze_solving)
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.image_count = 1

        
        # Creating objects for each stage of the robot navigation
        self.bot_localizer_mapper = None
        self.bot_localizer_mapper_kitchen = None
        self.bot_localizer_mapper_bedroom = None
        self.bot_pathplanner = None

        # Subscrbing to receive the robot pose in simulation
        #self.pose_subscriber = self.create_subscription(Odometry,'/odom',self.bot_motionplanner.get_pose,10)
        self.vel_subscriber = self.create_subscription(Odometry,'/odom',self.get_pose,10)
        self.bot_speed = 0
        self.bot_turning = 0
        self.yaw = 0
        self.desired_yaw = 0
        self.left = False
        self.right = False
        self.temp_goal = (0,0)

        self.sat_view = None
        self.place = None
               
    def get_video_feed_cb(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view = frame

    def get_video_feed_cb_kitchen(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view_kitchen = frame
    
    def get_video_feed_cb_bedroom(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view_bedroom = frame

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z 
  
    def get_pose(self,data):

        # We get the bot_turn_angle in simulation Using same method as Gotogoal.py
        quaternions = data.pose.pose.orientation
        (roll,pitch,yaw)=self.euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
        yaw_deg = degrees(yaw)
        self.yaw = yaw_deg

        # [Maintaining the Consistency in Angle Range]
        if (yaw_deg>0):
            self.yaw = yaw_deg
        else:
            # -160 + 360 = 200, -180 + 360 = 180 . -90 + 360 = 270
            self.yaw = yaw_deg + 360
        
        #              Bot Rotation 
        #      (OLD)        =>      (NEW) 
        #   [-180,180]             [0,360]

    
    def depthToCV8UC1(self, float_img):
        # Process images
        mono8_img = np.zeros_like(float_img, dtype=np.uint8)
        cv2.convertScaleAbs(float_img, mono8_img, alpha=10, beta=0.0)
        return mono8_img
    
    def process_data_depth(self, data):
      float_view = self.bridge.imgmsg_to_cv2(data, '8UC1')
      self.bot_view_depth = self.depthToCV8UC1(float_view)

    def process_data_raw(self, data):
      self.bot_view_raw = self.bridge.imgmsg_to_cv2(data, 'bgr8')


    def get_bot_speed(self,data):
        self.bot_speed = (data.twist.twist.linear.x)
        if self.bot_speed<0.1:
            self.bot_speed = 0.05

        self.bot_turning = data.twist.twist.angular.z

    def motion_stop(self):
        pics = self.bot_localizer_mapper.mapping_update(self.temp_goal) 
        cv2.imshow("map", pics)
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0

        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(self.vel_msg)

    def motion_rotate(self):
        pics = self.bot_localizer_mapper.mapping_update(self.temp_goal) 
        cv2.imshow("map", pics)
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0

        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.15
        self.velocity_publisher.publish(self.vel_msg)
        cv2.waitKey(1)
    
    def motion_forward(self):

        pics = self.bot_localizer_mapper.mapping_update(self.temp_goal) 
        cv2.imshow("map", pics)            
        self.vel_msg.linear.x = 0.1
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0

        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(self.vel_msg)
        cv2.waitKey(1)
              
    def update_desired_yaw(self, current_pos, desired_pos):
        if (desired_pos[0] - current_pos[0] == -1) and (desired_pos[1] - current_pos[1] == 0):
            self.desired_yaw = 0
        elif (desired_pos[0] - current_pos[0] == 1) and (desired_pos[1] - current_pos[1] == 0):
            self.desired_yaw = 180
        elif (desired_pos[0] - current_pos[0] == 0) and (desired_pos[1] - current_pos[1] == 1):
            self.desired_yaw = 270
        elif (desired_pos[0] - current_pos[0] == 0) and (desired_pos[1] - current_pos[1] == -1):
            self.desired_yaw = 90
        elif (desired_pos[0] - current_pos[0] == -1) and (desired_pos[1] - current_pos[1] == 1):
            self.desired_yaw = 315
        elif (desired_pos[0] - current_pos[0] == -1) and (desired_pos[1] - current_pos[1] == -1):
            self.desired_yaw = 45
        elif (desired_pos[0] - current_pos[0] == 1) and (desired_pos[1] - current_pos[1] == -1):
            self.desired_yaw = 135
        elif (desired_pos[0] - current_pos[0] == 1) and (desired_pos[1] - current_pos[1] == 1):
            self.desired_yaw = 225

    def maze_solving(self):
        cv2.imshow('raw', self.bot_view_raw)
        cv2.imshow('depth',self.bot_view_depth)

        frame = self.sat_view
        frame_kitchen = self.sat_view_kitchen
        frame_bedroom = self.sat_view_bedroom
        cv2.imshow("original", self.sat_view)

        self.bot_localizer_mapper = Map(frame)
        self.bot_localizer_mapper_kitchen = Map(frame_kitchen)
        self.bot_localizer_mapper_bedroom = Map(frame_bedroom)

        map = self.bot_localizer_mapper.mapping_update(self.temp_goal)
        map_kitchen = self.bot_localizer_mapper_kitchen.mapping_update(self.temp_goal)
        map_bedroom = self.bot_localizer_mapper_bedroom.mapping_update(self.temp_goal)

        #cv2.imshow("map", map)
        #cv2.imshow("map kitchen", map_kitchen)
        #cv2.imshow("map bedroom", map_bedroom)
        
        if self.bot_localizer_mapper.detected == True:
            self.place = 'living_room'
            cv2.imshow("map", map)
        if self.bot_localizer_mapper_kitchen.detected == True:
            self.place = 'kitchen'
            cv2.imshow("map kitchen", map_kitchen)
        if self.bot_localizer_mapper_bedroom.detected == True:
            self.place = 'bedroom' 
            cv2.imshow("map bedroom", map_bedroom)
        
        cv2.waitKey(1) 


        '''
        self.bot_pathplanner = Dijkstra(self.bot_localizer_mapper.grid_data, self.bot_localizer_mapper.x_size, self.bot_localizer_mapper.y_size)
        path = self.bot_pathplanner.perform((3, 23))
        #print(path)
        if path != None:
            if len(path) != 0:
                if self.bot_pathplanner.reached == False:
                    self.temp_goal = path.pop()
                    self.update_desired_yaw(self.bot_localizer_mapper.robot_pos,self.temp_goal)
                    print(self.yaw, self.desired_yaw , self.bot_localizer_mapper.robot_pos, self.temp_goal)
                    if abs(self.desired_yaw - self.yaw) > 5:
                        self.motion_rotate()
                        #print("rotate to ", self.desired_yaw)
                    else:
                        self.motion_forward()
                else:
                    self.motion_stop()
                
        '''
        


def main(args =None):
    rclpy.init()
    node_obj =maze_solver()
    rclpy.spin(node_obj)
    rclpy.shutdown()


if __name__ == '__main__':
    main()