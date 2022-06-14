#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Path
from functools import partial

class GoalManager:
    def __init__(self):
        self.goal_list_ = np.zeros((0, 2))
        self.claimed_goals_ = np.zeros((0, 2))
        self.other_claimed_goals_ = {}
        self.last_stamp_other_ = {}
        self.in_progress_ = False
        self.current_goal_ = None

        self.target_goals_sub_ = rospy.Subscriber("~target_goals", PoseArray, 
                self.target_goals_cb)
        self.navigate_status_sub_ = rospy.Subscriber("~navigate_status", Bool, 
                self.navigate_status_cb)

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

        self.target_goals_msg_ = PoseArray()
        self.target_goals_pub_ = rospy.Publisher("~claimed_goals", PoseArray, queue_size=1)
        self.goal_viz_pub_ = rospy.Publisher("~goal_viz", Image, queue_size=1)

        self.check_for_new_goal_timer_ = rospy.Timer(rospy.Duration(5), self.check_for_new_goal)
    
    def other_robot_cb(self, goals, take_pri, robot):
        rospy.loginfo(f"{self.this_robot_} got goals from {robot}")
        if goals.header.stamp.to_nsec() <= self.last_stamp_other_[robot]:
            return
        self.last_stamp_other_[robot] = goals.header.stamp.to_nsec()
        preempted = False
        self.other_claimed_goals_[robot] = np.zeros((0, 2))
        for pose in goals.poses:
            last_pt = np.array([pose.position.x, pose.position.y])
            self.other_claimed_goals_[robot] = np.vstack([self.other_claimed_goals_[robot], last_pt])
            if self.in_progress_ and self.current_goal_ is not None:
               dist = np.linalg.norm(self.current_goal_ - last_pt)
               if dist < 2 * self.region_size_ / self.scale_ and not take_pri:
                   preempted = True
        rospy.loginfo(self.other_claimed_goals_)

        if preempted:
            # report not successful, but only transmit if new goal found
            if len(self.target_goals_msg_.poses) > 0:
                self.target_goals_msg_.poses.pop()

            rospy.loginfo("Other robot with higher priority has goal")
            # preempt
            self.in_progress_ = False
            # manually trigger looking for new goal
            self.check_for_new_goal()

    def target_goals_cb(self):
        pass

    def get_all_other_goals(self):
        other_goals = np.zeros((0, 2))
        for robot_goals in self.other_claimed_goals_.values():
            other_goals = np.vstack([other_goals, robot_goals])
        return other_goals

    def check_for_new_goal(self, timer=None):
        if not self.in_progress_:
            selected_goal = self.choose_goal(blob_centers)
            if selected_goal is not None:
                rospy.loginfo("Found goal, executing plan")
                #self.planner_node_.pub_plan(goal_plan)
                self.current_goal_ = selected_goal
                self.in_progress_ = True

                # going to goal
                goal_pose = Pose()
                goal_pose.position.x = selected_goal[0]
                goal_pose.position.y = selected_goal[1]
                goal_pose.orientation.w = 1
                self.target_goals_msg_.header.stamp = rospy.Time.now()
                self.target_goals_msg_.poses.append(goal_pose)
                self.target_goals_pub_.publish(self.target_goals_msg_)
            else:
                rospy.logwarn("Cannot find any path to ROIs")

    def choose_goal(self, goals):
        last_pose = self.planner_node_.last_pose_
        if last_pose is None:
            rospy.logwarn("No Odometry Received Yet")
            return None

        best_cost = np.inf
        best_goal = None
        all_claimed_goals = np.vstack([self.get_all_other_goals(), self.claimed_goals_])
        for goal in goals:
            if all_claimed_goals.shape[0] > 0:
                dists_from_existing_goals = np.linalg.norm(all_claimed_goals - goal, axis=1)
                if np.min(dists_from_existing_goals) < self.region_size_ / self.scale_:
                    # Another goal is close, ignore
                    continue

            # Approximate
            cost = np.linalg.norm(last_pose - goal)
            if cost < best_cost:
                best_goal = goal
                best_cost = cost

        return best_goal

    def navigate_status_cb(self, status_msg):
        if self.current_goal_ is not None:
            # Add to visited targets
            # Do this regardless of the plan success, because we don't want to try to
            # plan to this target again and get stuck
            self.claimed_goals_ = np.vstack([self.claimed_goals_, self.last_plan_[-1]])
            self.current_goal_ = None

            if not status_msg.data:
                # did not get to goal successfully
                self.target_goals_msg_.header.stamp = rospy.Time.now()
                if len(self.target_goals_msg_.poses) > 0:
                    self.target_goals_msg_.poses.pop()
                self.target_goals_pub_.publish(self.target_goals_msg_)

        self.in_progress_ = False
        # Will now check for new goal when timer triggers

if __name__ == '__main__':
    rospy.init_node('goal_manager')
    gm = GoalManager()
    rospy.spin()

