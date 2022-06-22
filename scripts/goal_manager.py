#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
import actionlib
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool, ColorRGBA
from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped, Point
from spomp.msg import GlobalNavigateAction, GlobalNavigateGoal, GlobalNavigateResult
from nav_msgs.msg import Path
from functools import partial

class GoalManager:
    def __init__(self):
        self.goal_list_ = np.zeros((0, 2))
        self.visited_goals_ = np.zeros((0, 2))
        self.failed_goals_ = np.zeros((0, 2))
        self.other_claimed_goals_ = {}
        self.last_stamp_other_ = {}
        self.in_progress_ = False
        self.current_goal_ = None
        self.start_loc_ = None
        self.current_loc_ = None

        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.min_goal_dist_m_ = rospy.get_param("~min_goal_dist_m", default=5)

        self.add_goal_point_sub_ = rospy.Subscriber("~add_goal_point", PointStamped, 
                self.add_goal_point_cb)
        self.target_goals_sub_ = rospy.Subscriber("~target_goals", PoseArray, 
                self.target_goals_cb)
        self.navigate_status_sub_ = rospy.Subscriber("~navigate_status", Bool, 
                self.navigate_status_cb)
        self.pose_sub_ = rospy.Subscriber("~pose", PoseStamped, self.pose_cb)

        # Subscribers for goals of other robots
        robots = rospy.get_param("~robot_list").split(',')
        priorities = range(len(robots))
        this_robot = rospy.get_param("~this_robot")
        self.this_robot_ = this_robot
        this_priority = int(priorities[robots.index(this_robot)])
        self.other_robot_subs_ = []
        for robot, priority in zip(robots, priorities):
            if this_robot != robot:
                rospy.loginfo(f"{self.this_robot_} initializing other robot {robot}")
                self.last_stamp_other_[robot] = 0
                self.other_claimed_goals_[robot] = np.zeros((0, 2))
                bound_cb = partial(self.other_robot_cb, take_pri=int(priority) > this_priority, robot=robot)
                self.other_robot_subs_.append(rospy.Subscriber(robot+"/goal_manager/claimed_goals", PoseArray, bound_cb))

        self.claimed_goals_msg_ = PoseArray()
        self.claimed_goals_pub_ = rospy.Publisher("~claimed_goals", PoseArray, queue_size=1)
        self.goal_viz_pub_ = rospy.Publisher("~goal_viz", Marker, queue_size=1)
        self.navigate_client_ = actionlib.SimpleActionClient('/spomp_global/navigate', GlobalNavigateAction)

        rospy.loginfo("Waiting for spomp action server...")
        self.navigate_client_.wait_for_server()
        rospy.loginfo("Action server found")

        self.check_for_new_goal_timer_ = rospy.Timer(rospy.Duration(5), self.check_for_new_goal)
    
    def other_robot_cb(self, goals, take_pri, robot):
        rospy.loginfo(f"{self.this_robot_} got goals from {robot}")
        if goals.header.stamp.to_nsec() <= self.last_stamp_other_[robot]:
            rospy.loginfo("Message from other robot was old")
            return
        self.last_stamp_other_[robot] = goals.header.stamp.to_nsec()
        preempted = False
        self.other_claimed_goals_[robot] = np.zeros((0, 2))
        for pose in goals.poses:
            goal_pt = np.array([pose.position.x, pose.position.y])
            self.other_claimed_goals_[robot] = np.vstack([self.other_claimed_goals_[robot], goal_pt])
            if self.in_progress_ and self.current_goal_ is not None:
                dist = np.linalg.norm(self.current_goal_ - goal_pt)
                rospy.loginfo(dist)
                if dist < self.min_goal_dist_m_ and not take_pri:
                    rospy.loginfo("Preempted")
                    preempted = True
                else:
                    rospy.loginfo("Not preempted")
        rospy.loginfo(self.other_claimed_goals_)

        if preempted:
            # report not successful, but only transmit if new goal found
            if len(self.claimed_goals_msg_.poses) > 0:
                self.claimed_goals_msg_.poses.pop()

            rospy.loginfo("Other robot with higher priority has goal")
            # preempt
            self.in_progress_ = False
            self.current_goal_ = None
            # manually trigger looking for new goal
            self.check_for_new_goal()

        self.visualize()

    def add_goal_point_cb(self, goal_msg):
        try:
            trans_goal = self.tf_buffer_.transform(goal_msg, "map")
        except Exception as ex:
            rospy.logwarn(f"Cannot transform goals: {ex}")
            return

        goal_pt = np.array([trans_goal.point.x, trans_goal.point.y])

        # This is the simple interface, so do allow to manually add any goal
        self.goal_list_ = np.vstack([self.goal_list_, goal_pt])
        self.visualize()

    def target_goals_cb(self, goals_msg):
        goal_msg = PoseStamped()
        goal_msg.header = goals_msg.header
        for goal_pose in goals_msg.poses:
            try:
                # Have to transform a PoseStamped, not a PoseArray
                goal_msg.pose = goal_pose
                trans_goal = self.tf_buffer_.transform(goal_msg, "map")
                goal_pt = np.array([trans_goal.pose.position.x, trans_goal.pose.position.y])

                if self.goal_list_.shape[0] > 0:
                    # Don't allow duplicate goals
                    dists_from_existing_goals = np.linalg.norm(self.goal_list_ - goal_pt, axis=1)
                    if np.min(dists_from_existing_goals) < self.min_goal_dist_m_:
                        continue

                self.goal_list_ = np.vstack([self.goal_list_, goal_pt])
            except Exception as ex:
                rospy.logwarn(f"Cannot transform goal: {ex}")
                return

        self.visualize()

    def pose_cb(self, pose_msg):
        try:
            trans_pose = self.tf_buffer_.transform(pose_msg, "map")
        except Exception as ex:
            rospy.logwarn(f"Cannot transform pose: {ex}")
            return

        self.current_loc_ = np.array([trans_pose.pose.position.x,
                                      trans_pose.pose.position.y])

    def get_all_other_goals(self):
        other_goals = np.zeros((0, 2))
        for robot_goals in self.other_claimed_goals_.values():
            other_goals = np.vstack([other_goals, robot_goals])
        return other_goals

    def check_for_new_goal(self, timer=None):
        if not self.in_progress_:
            selected_goal = self.choose_goal()
            if selected_goal is not None:
                rospy.loginfo("Found goal, executing plan")
                self.current_goal_ = selected_goal
                if self.start_loc_ is None:
                    # save start position
                    self.start_loc_ = self.current_loc_
                self.in_progress_ = True

                # going to goal
                goal_pose = Pose()
                goal_pose.position.x = selected_goal[0]
                goal_pose.position.y = selected_goal[1]
                goal_pose.orientation.w = 1
                self.claimed_goals_msg_.header.stamp = rospy.Time.now()
                self.claimed_goals_msg_.header.frame_id = "map"
                self.claimed_goals_msg_.poses.append(goal_pose)
                self.claimed_goals_pub_.publish(self.claimed_goals_msg_)

                cur_goal_msg = GlobalNavigateGoal()
                cur_goal_msg.goal.header = self.claimed_goals_msg_.header
                cur_goal_msg.goal.pose = goal_pose
                self.navigate_client_.send_goal(cur_goal_msg, done_cb=self.navigate_status_cb)
            else:
                rospy.logwarn("Cannot find any path to goals")

                # go home
                if self.start_loc_ is not None:
                    cur_goal_msg = GlobalNavigateGoal()
                    cur_goal_msg.header.stamp = rospy.Time.now()
                    cur_goal_msg.header.frame_id = "map"
                    cur_goal_msg.goal.pose.position.x = self.start_loc_[0]
                    cur_goal_msg.goal.pose.position.y = self.start_loc_[1]
                    cur_goal_msg.goal.pose.orientation.w = 1
                    # don't send with callback, since we don't care about status
                    self.navigate_client_.send_goal(cur_goal_msg)

                # no other goals available, so let's try failed ones again
                self.failed_goals_ = np.zeros((0, 2))
        self.visualize()

    def choose_goal(self):
        if self.current_loc_ is None:
            rospy.logwarn("No Odometry Received Yet")
            return None

        best_cost = np.inf
        best_goal = None
        all_claimed_goals = np.vstack([self.get_all_other_goals(), 
            self.visited_goals_, self.failed_goals_])
        for goal in self.goal_list_:
            if all_claimed_goals.shape[0] > 0:
                dists_from_existing_goals = np.linalg.norm(all_claimed_goals - goal, axis=1)
                if np.min(dists_from_existing_goals) < self.min_goal_dist_m_:
                    # Another goal is close, ignore
                    continue

            # Approximate
            cost = np.linalg.norm(self.current_loc_- goal)
            if cost < best_cost:
                best_goal = goal
                best_cost = cost

        return best_goal

    def navigate_status_cb(self, status_msg, result_msg):
        if self.current_goal_ is not None:
            # Add to visited targets
            if result_msg.status == GlobalNavigateResult.SUCCESS:
                self.visited_goals_ = np.vstack([self.visited_goals_, self.current_goal_[None,:]])

            if result_msg.status == GlobalNavigateResult.TIMEOUT or \
               result_msg.status == GlobalNavigateResult.NO_PATH:
                rospy.logerr("Failed to get to goal")
                # did not get to goal successfully
                self.failed_goals_ = np.vstack([self.failed_goals_, self.current_goal_[None,:]])
                self.claimed_goals_msg_.header.stamp = rospy.Time.now()
                if len(self.claimed_goals_msg_.poses) > 0:
                    self.claimed_goals_msg_.poses.pop()
                self.claimed_goals_pub_.publish(self.claimed_goals_msg_)

            self.current_goal_ = None

        # Will now check for new goal when timer triggers
        self.in_progress_ = False
        self.visualize()

    def visualize(self):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = "map"
        marker_msg.ns = "robot_goals"
        marker_msg.id = 0
        marker_msg.type = Marker.SPHERE_LIST
        marker_msg.action = Marker.ADD
        marker_msg.pose.position.z = 1
        marker_msg.pose.orientation.w = 1
        marker_msg.scale.x = 5
        marker_msg.scale.y = 5
        marker_msg.scale.z = 5
        marker_msg.color.a = 1

        visualized_goals = np.zeros((0, 2))

        # helper function
        def add_goal_to_viz(goal, color):
            nonlocal visualized_goals
            nonlocal marker_msg
            if visualized_goals.shape[0] > 0:
                dists_from_vis_goals = np.linalg.norm(visualized_goals - goal, axis=1)
                if np.min(dists_from_vis_goals) < self.min_goal_dist_m_:
                    return
            visualized_goals = np.vstack([visualized_goals, goal])

            pt_msg = Point()
            color_msg = ColorRGBA()
            color_msg.a = 1
            pt_msg.x = goal[0]
            pt_msg.y = goal[1]
            color_msg.r = color[0]
            color_msg.g = color[1]
            color_msg.b = color[2]
            marker_msg.points.append(pt_msg)
            marker_msg.colors.append(color_msg)

        for goal in self.visited_goals_:
            add_goal_to_viz(goal, (0, 1, 0))

        for goal in self.failed_goals_:
            add_goal_to_viz(goal, (1, 0, 0))

        for goal in self.get_all_other_goals():
            add_goal_to_viz(goal, (0.5, 1, 0.5))

        for goal in self.goal_list_:
            add_goal_to_viz(goal, (0.8, 0.8, 0.8))

        self.goal_viz_pub_.publish(marker_msg)        

if __name__ == '__main__':
    rospy.init_node('goal_manager')
    gm = GoalManager()
    rospy.spin()

