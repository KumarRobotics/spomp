#!/usr/bin/env python3
import threading
import numpy as np
import rospy
#import tf2_ros
#import tf2_geometry_msgs
import actionlib
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool, ColorRGBA
from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped, Point, PoseWithCovarianceStamped
from spomp.msg import GlobalNavigateAction, GlobalNavigateGoal, GlobalNavigateResult, ClaimedGoal, ClaimedGoalArray
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
        self.rtls_ = False
        self.current_goal_ = None
        self.start_loc_ = None
        self.current_loc_ = None
        self.lock_ = threading.Lock()

        #self.tf_buffer_ = tf2_ros.Buffer()
        #self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.min_goal_dist_m_ = rospy.get_param("~min_goal_dist_m", default=5)

        self.add_goal_point_sub_ = rospy.Subscriber("~add_goal_point", PoseWithCovarianceStamped, 
                self.add_goal_point_cb)
        self.target_goals_sub_ = rospy.Subscriber("~target_goals", PoseArray, 
                self.target_goals_cb)
        self.navigate_status_sub_ = rospy.Subscriber("~navigate_status", Bool, 
                self.navigate_status_cb)
        self.pose_sub_ = rospy.Subscriber("~pose", PoseWithCovarianceStamped, self.pose_cb)

        # Subscribers for goals of other robots
        robots = rospy.get_param("~robot_list").split(',')
        priorities = range(len(robots))
        this_robot = rospy.get_param("~this_robot")
        self.this_robot_ = this_robot

        rospy.loginfo("\033[32m[GoalManager] \n" +
                f"======== Configuration ========\n" +
                f"min_goal_dist_m: {self.min_goal_dist_m_}\n" +
                f"robot_list: {robots}\n" +
                f"this_robot: {self.this_robot_}\n" +
                f"====== End Configuration ======\033[0m")

        this_priority = int(priorities[robots.index(this_robot)])
        self.other_robot_subs_ = []
        for robot, priority in zip(robots, priorities):
            if this_robot != robot:
                rospy.loginfo(f"[GoalManager] {self.this_robot_} initializing other robot {robot}")
                self.last_stamp_other_[robot] = 0
                self.other_claimed_goals_[robot] = np.zeros((0, 2))
                bound_cb = partial(self.other_robot_cb, take_pri=int(priority) > this_priority, robot=robot)
                self.other_robot_subs_.append(rospy.Subscriber(robot+"/goal_manager/claimed_goals", ClaimedGoalArray, bound_cb))

        self.claimed_goals_msg_ = ClaimedGoalArray()
        self.claimed_goals_pub_ = rospy.Publisher("~claimed_goals", ClaimedGoalArray, queue_size=10)
        self.goal_viz_pub_ = rospy.Publisher("~goal_viz", Marker, queue_size=10)
        self.navigate_client_ = actionlib.SimpleActionClient('spomp_global/navigate', GlobalNavigateAction)

        rospy.loginfo("[GoalManager] Waiting for spomp action server...")
        self.navigate_client_.wait_for_server()
        rospy.loginfo("[GoalManager] Action server found")

        # offset frequencies to reduce chance of collision
        self.check_for_new_goal_timer_ = rospy.Timer(rospy.Duration(5 + 1*this_priority), self.check_for_new_goal)
    
    def other_robot_cb(self, goals, take_pri, robot):
        self.lock_.acquire()

        rospy.loginfo(f"[GoalManager] {self.this_robot_} got goals from {robot}")
        if goals.header.stamp.to_nsec() <= self.last_stamp_other_[robot]:
            rospy.loginfo("[GoalManager] Message from other robot was old")
            self.lock_.release()
            return
        self.last_stamp_other_[robot] = goals.header.stamp.to_nsec()
        preempted = False
        self.other_claimed_goals_[robot] = np.zeros((0, 2))
        for goal in goals.goals:
            goal_pt = np.array([goal.position.x, goal.position.y])
            self.other_claimed_goals_[robot] = np.vstack([self.other_claimed_goals_[robot], goal_pt])
            if self.in_progress_ and self.current_goal_ is not None:
                dist = np.linalg.norm(self.current_goal_ - goal_pt)
                rospy.loginfo(f"[GoalManager] Other goal dist: {dist}")
                if dist < self.min_goal_dist_m_ and (not take_pri or goal.status == ClaimedGoal.VISITED):
                    # preempt if we are not priority, or if goal has already been visited
                    rospy.loginfo("[GoalManager] Preempted")
                    preempted = True
                else:
                    rospy.loginfo("[GoalManager] Not preempted")
                    rospy.loginfo(f"[GoalManager] Other goals: {self.other_claimed_goals_}")

        if preempted:
            # report not successful
            self.claimed_goals_msg_.header.stamp = rospy.Time.now()
            if len(self.claimed_goals_msg_.goals) > 0:
                self.claimed_goals_msg_.goals.pop()
            self.claimed_goals_pub_.publish(self.claimed_goals_msg_)

            rospy.loginfo("[GoalManager] Other robot with higher priority has goal")
            # preempt
            self.in_progress_ = False
            self.current_goal_ = None
            # manually trigger looking for new goal
            self.lock_.release()
            self.check_for_new_goal(None, True)
            self.lock_.acquire()

        self.visualize()
        self.lock_.release()

    def add_goal_point_cb(self, goal_msg):
        try:
            trans_goal = goal_msg
            #trans_goal = self.tf_buffer_.transform(goal_msg, "map")
        except Exception as ex:
            rospy.logwarn(f"[GoalManager] Cannot transform goals: {ex}")
            return
        self.lock_.acquire()

        goal_pt = np.array([trans_goal.pose.pose.position.x, trans_goal.pose.pose.position.y])

        # This is the simple interface, so do allow to manually add any goal
        self.goal_list_ = np.vstack([self.goal_list_, goal_pt])
        self.visualize()
        self.lock_.release()

    def target_goals_cb(self, goals_msg):
        self.lock_.acquire()
        goal_msg = PoseStamped()
        goal_msg.header = goals_msg.header
        for goal_pose in goals_msg.poses:
            try:
                # Have to transform a PoseStamped, not a PoseArray
                goal_msg.pose = goal_pose
                trans_goal = goal_msg
                #trans_goal = self.tf_buffer_.transform(goal_msg, "map")
                goal_pt = np.array([trans_goal.pose.position.x, trans_goal.pose.position.y])

                if self.goal_list_.shape[0] > 0:
                    # Don't allow duplicate goals
                    dists_from_existing_goals = np.linalg.norm(self.goal_list_ - goal_pt, axis=1)
                    if np.min(dists_from_existing_goals) < self.min_goal_dist_m_:
                        continue

                self.goal_list_ = np.vstack([self.goal_list_, goal_pt])
            except Exception as ex:
                rospy.logwarn(f"[GoalManager] Cannot transform goal: {ex}")
                self.lock_.release()
                return

        self.visualize()
        self.lock_.release()

    def pose_cb(self, pose_msg):
        trans_pose = pose_msg.pose
        #try:
        #    trans_pose = pose_msg
        #    #trans_pose = self.tf_buffer_.transform(pose_msg, "map")
        #except Exception as ex:
        #    rospy.logwarn(f"Cannot transform pose: {ex}")
        #    return
        self.lock_.acquire()

        self.current_loc_ = np.array([trans_pose.pose.position.x,
                                      trans_pose.pose.position.y])
        self.lock_.release()

    def get_all_other_goals(self):
        other_goals = np.zeros((0, 2))
        for robot_goals in self.other_claimed_goals_.values():
            other_goals = np.vstack([other_goals, robot_goals])
        return other_goals

    def check_for_new_goal(self, timer=None, force_rtls=False):
        self.lock_.acquire()
        if not self.in_progress_:
            selected_goal = self.choose_goal()
            if selected_goal is not None:
                rospy.loginfo("[GoalManager] Found goal, executing plan")
                self.current_goal_ = selected_goal
                if self.start_loc_ is None:
                    # save start position
                    self.start_loc_ = self.current_loc_
                self.in_progress_ = True

                # going to goal
                goal_pose = ClaimedGoal()
                goal_pose.position.x = selected_goal[0]
                goal_pose.position.y = selected_goal[1]
                goal_pose.status = ClaimedGoal.IN_PROGRESS
                self.claimed_goals_msg_.header.stamp = rospy.Time.now()
                self.claimed_goals_msg_.header.stamp.nsecs += 1
                self.claimed_goals_msg_.header.frame_id = "map"
                self.claimed_goals_msg_.goals.append(goal_pose)
                self.claimed_goals_pub_.publish(self.claimed_goals_msg_)

                cur_goal_msg = GlobalNavigateGoal()
                cur_goal_msg.force = False
                cur_goal_msg.goal.header = self.claimed_goals_msg_.header
                cur_goal_msg.goal.pose.position = goal_pose.position
                cur_goal_msg.goal.pose.orientation.w = 1
                self.rtls_ = False
                self.navigate_client_.send_goal(cur_goal_msg, done_cb=self.navigate_status_cb)
            else:
                rospy.logwarn("[GoalManager] Cannot find any path to goals")

                # go home
                if self.start_loc_ is not None and not self.rtls_:
                    if np.linalg.norm(self.start_loc_ - self.current_loc_) > 5 or force_rtls:
                        rospy.loginfo("[GoalManager] Returning to start")
                        self.rtls_ = True
                        cur_goal_msg = GlobalNavigateGoal()
                        cur_goal_msg.force = True
                        cur_goal_msg.goal.header.stamp = rospy.Time.now()
                        cur_goal_msg.goal.header.frame_id = "map"
                        cur_goal_msg.goal.pose.position.x = self.start_loc_[0]
                        cur_goal_msg.goal.pose.position.y = self.start_loc_[1]
                        cur_goal_msg.goal.pose.orientation.w = 1
                        # don't send with callback, since we don't care about status
                        self.navigate_client_.send_goal(cur_goal_msg, done_cb=self.navigate_status_cb)

                # no other goals available, so let's try failed ones again
                if not self.rtls_:
                    self.failed_goals_ = np.zeros((0, 2))
        self.visualize()
        self.lock_.release()

    def choose_goal(self):
        if self.current_loc_ is None:
            rospy.logwarn("[GoalManager] No Odometry Received Yet")
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
        self.lock_.acquire()
        self.rtls_ = False

        if self.current_goal_ is not None:
            # Add to visited targets
            self.claimed_goals_msg_.header.stamp = rospy.Time.now()
            # This is a somewhat hacky fix.  In simulation, rostime is discretized, and so 
            # when we call out for a new path and it can't be found, the time is the same.
            # Add a little time here to guarantee the stamp is later
            self.claimed_goals_msg_.header.stamp.nsecs += 2
            if result_msg.status == GlobalNavigateResult.SUCCESS:
                self.visited_goals_ = np.vstack([self.visited_goals_, self.current_goal_[None,:]])

                if len(self.claimed_goals_msg_.goals) > 0:
                    self.claimed_goals_msg_.goals[-1].status = ClaimedGoal.VISITED
                self.claimed_goals_pub_.publish(self.claimed_goals_msg_)
            elif result_msg.status == GlobalNavigateResult.FAILED or \
               result_msg.status == GlobalNavigateResult.TIMEOUT or \
               result_msg.status == GlobalNavigateResult.NO_PATH:
                reason = "other"
                if result_msg.status == GlobalNavigateResult.FAILED:
                    reason = "failed"
                elif result_msg.status == GlobalNavigateResult.TIMEOUT:
                    reason = "timeout"
                elif result_msg.status == GlobalNavigateResult.NO_PATH:
                    reason = "no_path"
                rospy.logerr(f"[GoalManager] Failed to get to goal: {reason}")

                # did not get to goal successfully
                self.failed_goals_ = np.vstack([self.failed_goals_, self.current_goal_[None,:]])
                if len(self.claimed_goals_msg_.goals) > 0:
                    self.claimed_goals_msg_.goals.pop()
                self.claimed_goals_pub_.publish(self.claimed_goals_msg_)

            self.current_goal_ = None

        # Will now check for new goal when timer triggers
        self.in_progress_ = False
        self.visualize()
        self.lock_.release()

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
            add_goal_to_viz(goal, (1, 1, 0))

        for goal in self.goal_list_:
            add_goal_to_viz(goal, (0.8, 0.8, 0.8))

        self.goal_viz_pub_.publish(marker_msg)        

if __name__ == '__main__':
    rospy.init_node('goal_manager')
    gm = GoalManager()
    rospy.spin()

