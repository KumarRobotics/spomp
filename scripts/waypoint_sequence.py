#!/usr/bin/env python3

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

def ros2np(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])

class WaypointSequence():
    def __init__(self):
        self.path_ = np.zeros([0, 2])
        self.next_waypt_ind_ = 0
        self.in_progress_ = False

        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)
        self.path_file_ = rospy.get_param('~path_file', './default.npy')

        self.waypt_sub_ = rospy.Subscriber('~waypt_goal', PointStamped, self.waypt_cb)
        self.pose_sub_ = rospy.Subscriber('~pose', PoseStamped, self.pose_cb)

        self.local_waypt_pub_ = rospy.Publisher('~local_waypt', PoseStamped, queue_size=1)
        self.path_viz_pub_ = rospy.Publisher('~path_viz', Path, queue_size=1)
        self.waypt_viz_pub_ = rospy.Publisher('~waypt_viz', Marker, queue_size=1)

    def save_path(self):
        np.save(self.path_file_, self.path_)

    def load_path(self):
        if path.exists(self.path_file_):
            rospy.loginfo("Loading path " + self.path_file_)
            self.path_ = np.load(self.path_file_)
        else:
            rospy.loginfo("No saved path found at " + self.path_file_)

    def pub_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.poses = []
        
        marker_msg = Marker()
        marker_msg.header = path_msg.header
        marker_msg.ns = 'waypt'
        marker_msg.type = Marker.SPHERE_LIST
        marker_msg.pose.orientation.w = 1 #unity pose
        marker_msg.scale.x = 3
        marker_msg.scale.y = 3
        for ind, pt in enumerate(self.path_):
            pt_msg = PoseStamped()
            pt_msg.header = path_msg.header
            #just for viz, so orientation doesn't matter
            pt_msg.pose.orientation.w = 1
            pt_msg.pose.position.x = pt[0]
            pt_msg.pose.position.y = pt[1]
            pt_msg.pose.position.z = 0
            path_msg.poses.append(pt_msg)

            pt_msg = Point()
            pt_msg.x = pt[0]
            pt_msg.y = pt[1]
            pt_msg.z = 0
            marker_msg.points.append(pt_msg)
            color = ColorRGBA()            
            if ind == self.next_waypt_ind_:
                color.r = 0
                color.g = 1
                color.b = 0
                color.a = 1
            else:
                color.r = 1
                color.g = 0
                color.b = 0
                color.a = 1
            marker_msg.colors.append(color)

        self.path_viz_pub_.publish(path_msg)
        self.waypt_viz_pub_.publish(marker_msg)

    def waypt_cb(self, waypt):
        rospy.loginfo("Got waypt")
        waypt_map = waypt
        if waypt.header.frame_id != 'map':
            try:
                waypt_to_map_trans = self.tf_buffer_.lookup_transform('map', waypt.header.frame_id, rospy.Time(0))
                waypt_map = tf2_geometry_msgs.do_transform_point(waypt, waypt_to_map_trans);
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Cannot transform waypt into map frame")
                return

        pt = np.array([waypt_map.point.x, waypt_map.point.y])
        self.path_ = np.concatenate((self.path_, pt[None,:]), axis=0)
        self.save_path()
        self.pub_path()

    def pose_cb(self, pose):
        if self.next_waypt_ind_ == 0 and self.path_.shape[0] > 0 and not self.in_progress_:
            # start mission if waypoint available
            self.pub_local_goal()

        if self.next_waypt_ind_ < self.path_.shape[0] and self.in_progress_:
            pose_map = pose
            if pose.header.frame_id != 'map':
                try:
                    pose_to_map_trans = self.tf_buffer_.lookup_transform('map', pose.header.frame_id, rospy.Time(0))
                    pose_map = tf2_geometry_msgs.do_transform_pose(pose, pose_to_map_trans);
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logerr("Cannot transform pose into map frame")
                    return

            pose_np = ros2np(pose_map.pose)
            if np.linalg.norm(pose_np[:2] - self.path_[self.next_waypt_ind_, :]) < 5:
                self.next_waypt_ind_ += 1
                if self.next_waypt_ind_ < self.path_.shape[0]:
                    self.pub_local_goal()
                else:
                    rospy.loginfo("Path complete")
                    self.in_progress_ = False

        self.pub_path()

    def pub_local_goal(self):
        next_goal = PoseStamped()
        next_goal.header.frame_id = 'map'
        next_goal.header.stamp = rospy.Time()
        next_goal.pose.position.x = self.path_[self.next_waypt_ind_, 0]
        next_goal.pose.position.y = self.path_[self.next_waypt_ind_, 1]
        next_goal.pose.orientation.w = 1
        self.in_progress_ = True
        self.local_waypt_pub_.publish(next_goal)

if __name__=='__main__':
    rospy.init_node('waypoint_sequence')
    ws = WaypointSequence()
    rospy.spin()
