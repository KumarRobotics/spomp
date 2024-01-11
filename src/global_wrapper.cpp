#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <grid_map_comp/grid_map_comp.hpp>
#include "spomp/global_wrapper.h"
#include "spomp/timer.h"
#include "spomp/rosutil.h"

namespace spomp {

std::string GlobalWrapper::odom_frame_{"odom"};
std::string GlobalWrapper::map_frame_{"map"};
std::vector<std::string> GlobalWrapper::other_robot_list_{};
std::string GlobalWrapper::this_robot_{"robot"};

/**
 * @brief The GlobalWrapper class
 *
 * The GlobalWrapper class is responsible for managing the global navigation module in the system.
 * It initializes the required publishers, subscribers, and service clients, as well as provides
 * functions for visualizing the graph and managing global navigation goals.
 */
    GlobalWrapper::GlobalWrapper(ros::NodeHandle& nh) :
  nh_(nh), 
  global_(createGlobal(nh)),
  it_(nh),
  tf_buffer_(),
  tf_listener_(tf_buffer_),
  global_navigate_as_(nh_, "navigate", false)
{
  // Publishers
  // Latch goal so it is received even if local planner hasn't started yet
  local_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("local_goal", 5, true);
  reachability_history_pub_ = nh_.advertise<LocalReachabilityArray>("reachability_history", 1, true);
  graph_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("graph_viz", 1, true);
  path_viz_pub_ = nh_.advertise<nav_msgs::Path>("path_viz", 1, true);
  map_img_viz_pub_ = it_.advertise("viz_img", 1);
  aerial_map_trav_viz_pub_ = it_.advertise("aerial_map_trav_viz", 1);

  visualizeGraph(ros::Time::now());
}

/****************************************************************************************
 * @brief Creates a Global object with the given parameters.
 *
 * The function first initializes various Params objects, such as g_params, tm_params,
 * tg_params, am_params, mlp_params, and wm_params, with the values obtained from
 * the ROS NodeHandle object nh using the nh.getParam() function.
 *
 * Next, the function parses the robot_list string obtained from nh.getParam() and fills
 * the other_robot_list_ vector after skipping the current robot name (this_robot_).
 *
 * Finally, the function prints the obtained parameter values to the console using the
 * ROS_INFO_STREAM macro.
 *
 * @param nh A reference to the ROS NodeHandle object.
 * @return A Global object with the given parameters.
 ****************************************************************************************/
    Global GlobalWrapper::createGlobal(ros::NodeHandle& nh) {
  Global::Params g_params{};
  TravMap::Params tm_params{};
  TravGraph::Params tg_params{};
  AerialMapInfer::Params am_params{};
  MLPModel::Params mlp_params{};
  WaypointManager::Params wm_params{};

  nh.getParam("world_config_path", tm_params.world_config_path);
  std::string robot_list_str;
  nh.getParam("robot_list", robot_list_str);
  nh.getParam("this_robot", this_robot_);

  // Parse robot list
  std::istringstream robots(robot_list_str);
  std::string robot;
  while (std::getline(robots, robot, ',')) {
    if (robot == this_robot_) continue;
    other_robot_list_.push_back(robot);
  }
  tm_params.num_robots = other_robot_list_.size() + 1;

  nh.getParam("G_max_num_recovery_reset", g_params.max_num_recovery_reset);
  nh.getParam("G_timeout_duration_s_per_m", g_params.timeout_duration_s_per_m);
  nh.getParam("G_replan_hysteresis", g_params.replan_hysteresis);
  nh.getParam("G_num_new_reach_to_pub", g_params.num_new_reach_to_pub);

  nh.getParam("TM_learn_trav", tm_params.learn_trav);
  nh.getParam("TM_uniform_node_sampling", tm_params.uniform_node_sampling);
  nh.getParam("TM_no_max_terrain_in_graph", tm_params.no_max_terrain_in_graph);
  nh.getParam("TM_max_hole_fill_size_m", tm_params.max_hole_fill_size_m);
  nh.getParam("TM_min_region_size_m", tm_params.min_region_size_m);
  nh.getParam("TM_vis_dist_m", tm_params.vis_dist_m);
  nh.getParam("TM_unvis_start_thresh", tm_params.unvis_start_thresh);
  nh.getParam("TM_unvis_stop_thresh", tm_params.unvis_stop_thresh);
  nh.getParam("TM_prune", tm_params.prune);
  nh.getParam("TM_max_prune_edge_dist_m", tm_params.max_prune_edge_dist_m);
  nh.getParam("TM_recover_reset_dist_m", tm_params.recover_reset_dist_m);
  nh.getParam("TM_reach_dist_thresh_m", tm_params.reach_dist_thresh_m);

  nh.getParam("TG_reach_node_max_dist_m", tg_params.reach_node_max_dist_m);
  nh.getParam("TG_trav_window_rad", tg_params.trav_window_rad);
  nh.getParam("TG_max_trav_discontinuity_m", tg_params.max_trav_discontinuity_m);
  nh.getParam("TG_num_edge_exp_before_mark", tg_params.num_edge_exp_before_mark);
  nh.getParam("TG_trav_edge_prob_trav", tg_params.trav_edge_prob_trav);

  nh.getParam("AM_inference_thread_period_ms", am_params.inference_thread_period_ms);
  nh.getParam("AM_trav_thresh", am_params.trav_thresh);
  nh.getParam("AM_not_trav_thresh", am_params.not_trav_thresh);
  nh.getParam("AM_not_trav_range_m", am_params.not_trav_range_m);
  nh.getParam("AM_max_frac_unknown", am_params.max_frac_unknown);

  nh.getParam("MLP_hidden_layer_size", mlp_params.hidden_layer_size);
  nh.getParam("MLP_regularization", mlp_params.regularization);

  nh.getParam("WM_waypoint_thresh_m", wm_params.waypoint_thresh_m);
  nh.getParam("WM_final_waypoint_thresh_m", wm_params.final_waypoint_thresh_m);
  nh.getParam("WM_shortcut_thresh_m", wm_params.shortcut_thresh_m);

  constexpr int width = 30;
  using namespace std;
  ROS_INFO_STREAM("\033[32m" << "[SPOMP-Global]" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(width) << "[ROS] world_config_path: " << tm_params.world_config_path << endl <<
    setw(width) << "[ROS] robot_list: " << robot_list_str << endl <<
    setw(width) << "[ROS] this_robot: " << this_robot_ << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] G_max_num_recovery_reset: " << g_params.max_num_recovery_reset << endl <<
    setw(width) << "[ROS] G_timeout_duration_s_per_m: " << g_params.timeout_duration_s_per_m << endl <<
    setw(width) << "[ROS] G_replan_hysteresis: " << g_params.replan_hysteresis << endl <<
    setw(width) << "[ROS] G_num_new_reach_to_pub: " << g_params.num_new_reach_to_pub << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] TM_learn_trav: " << tm_params.learn_trav << endl <<
    setw(width) << "[ROS] TM_uniform_node_sampling: " << tm_params.uniform_node_sampling << endl <<
    setw(width) << "[ROS] TM_no_max_terrain_in_graph: " << tm_params.no_max_terrain_in_graph << endl <<
    setw(width) << "[ROS] TM_max_hole_fill_size_m: " << tm_params.max_hole_fill_size_m << endl <<
    setw(width) << "[ROS] TM_min_region_size_m: " << tm_params.min_region_size_m << endl <<
    setw(width) << "[ROS] TM_vis_dist_m: " << tm_params.vis_dist_m << endl <<
    setw(width) << "[ROS] TM_unvis_start_thresh: " << tm_params.unvis_start_thresh << endl <<
    setw(width) << "[ROS] TM_unvis_stop_thresh: " << tm_params.unvis_stop_thresh << endl <<
    setw(width) << "[ROS] TM_prune: " << tm_params.prune << endl <<
    setw(width) << "[ROS] TM_max_prune_edge_dist_m: " << tm_params.max_prune_edge_dist_m << endl <<
    setw(width) << "[ROS] TM_recover_reset_dist_m: " << tm_params.recover_reset_dist_m << endl <<
    setw(width) << "[ROS] TM_reach_dist_thresh_m: " << tm_params.reach_dist_thresh_m << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] AM_inference_thread_period_ms: " << am_params.inference_thread_period_ms << endl <<
    setw(width) << "[ROS] AM_trav_thresh: " << am_params.trav_thresh << endl <<
    setw(width) << "[ROS] AM_not_trav_thresh: " << am_params.not_trav_thresh << endl <<
    setw(width) << "[ROS] AM_not_trav_range_m: " << am_params.not_trav_range_m << endl <<
    setw(width) << "[ROS] AM_max_frac_unknown: " << am_params.max_frac_unknown << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] MLP_hidden_layer_size: " << mlp_params.hidden_layer_size << endl <<
    setw(width) << "[ROS] MLP_regularization: " << mlp_params.regularization << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] TG_reach_node_max_dist_m: " << tg_params.reach_node_max_dist_m << endl <<
    setw(width) << "[ROS] TG_trav_window_rad: " << tg_params.trav_window_rad << endl <<
    setw(width) << "[ROS] TG_max_trav_discontinuity_m: " << tg_params.max_trav_discontinuity_m << endl <<
    setw(width) << "[ROS] TG_num_edge_exp_before_mark: " << tg_params.num_edge_exp_before_mark << endl <<
    setw(width) << "[ROS] TG_trav_edge_prob_trav: " << tg_params.trav_edge_prob_trav << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] WM_waypoint_thresh_m: " << wm_params.waypoint_thresh_m << endl <<
    setw(width) << "[ROS] WM_final_waypoint_thresh_m: " << wm_params.final_waypoint_thresh_m << endl <<
    setw(width) << "[ROS] WM_shortcut_thresh_m: " << wm_params.shortcut_thresh_m << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");

  return Global(g_params, tm_params, tg_params, am_params, mlp_params, wm_params);
}

/**
 * @brief Initializes the GlobalWrapper object.
 *
 * This function sets up the necessary subscribers and action server for the GlobalWrapper.
 * It also starts the ROS node spinner to handle callbacks.
 */
    void GlobalWrapper::initialize() {
  // Subscribers
  aerial_map_sub_ = nh_.subscribe("aerial_map", 5, 
      &GlobalWrapper::aerialMapCallback, this);
  pose_sub_ = nh_.subscribe("pose", 5, &GlobalWrapper::poseCallback, this);
  goal_sub_ = nh_.subscribe("goal_simple", 5, &GlobalWrapper::goalSimpleCallback, this);
  reachability_sub_ = nh_.subscribe("reachability", 5, 
      &GlobalWrapper::reachabilityCallback, this);

  // Start at 1 since this robot is 0
  int id = 1;
  for (const auto& robot : other_robot_list_) {
    other_robot_reachability_subs_.push_back(nh_.subscribe<LocalReachabilityArray>(
          "/" + ros::this_node::getNamespace() + "/" + robot + "/spomp_global/reachability_history", 5, 
          std::bind(&GlobalWrapper::otherReachabilityCallback, this, id, std::placeholders::_1)));
    ++id;
  }

  global_navigate_as_.registerGoalCallback(
      std::bind(&GlobalWrapper::globalNavigateGoalCallback, this));
  // This actually gets called by the action server as well when a new
  // goal is sent, which we don't want to do because sometimes goals are
  // sent just to check if we can get there
  //global_navigate_as_.registerPreemptCallback(
  //    std::bind(&GlobalWrapper::globalNavigatePreemptCallback, this));
  global_navigate_as_.start();

  ros::spin();
}

/**
 * @brief Callback function for receiving aerial map messages.
 *
 * This function is called whenever a new aerial map message is received. It processes the map message and updates the global map.
 *
 * @param map_msg The received aerial map message.
 */
    void GlobalWrapper::aerialMapCallback(
    const grid_map_msgs::GridMap::ConstPtr& map_msg) 
{
  if (map_msg->info.header.stamp.toNSec() <= last_map_stamp_) return;
  if (map_msg->info.length_x <= 0 || map_msg->info.length_y <= 0) return;

  ROS_DEBUG_STREAM("Got new map");
  last_map_stamp_ = map_msg->info.header.stamp.toNSec();

  cv::Mat sem_map_img;
  grid_map::GridMapComp::toImage(*map_msg, {"semantics", "", "char"}, sem_map_img);

  std::vector<cv::Mat> other_maps;
  other_maps.resize(2);
  grid_map::GridMapComp::toImage(*map_msg, {"color", "", "rgb"}, other_maps[0]);
  grid_map::GridMapComp::toImage(*map_msg, {"elevation", "", "float"}, other_maps[1]);

  global_.updateMap(sem_map_img,
      {map_msg->info.pose.position.x, map_msg->info.pose.position.y}, 
      other_maps);
  visualizeGraph(map_msg->info.header.stamp);
}

/**
 * @brief Callback function to process received pose messages and perform necessary actions.
 *
 * This function is the callback for the "pose" topic. It transforms the pose message to the map frame
 * if it is not already in the map frame. It then sets the state of the waypoint manager using the transformed pose.
 * It publishes the local goal using the timestamp from the received pose message.
 * If the state of the waypoint manager is "GOAL_REACHED", it cancels the local planner and, if using an action server,
 * sets the result as "SUCCESS" and marks the goal as reached.
 *
 * @param pose_msg The received pose message.
 */
    void GlobalWrapper::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
  auto pose_map_frame = *pose_msg;
  if (pose_map_frame.header.frame_id != map_frame_) {
    try {
      // Ask for most recent transform, treating the odom frame as fixed
      // This essentially tells tf to assume that the odom frame has not changed
      // since the last update, which is a valid assumption to make.
      pose_map_frame = tf_buffer_.transform(*pose_msg, map_frame_, ros::Time(0), 
          pose_msg->header.frame_id);
    } catch (tf2::TransformException& ex) {
      ROS_ERROR_STREAM("Cannot transform pose to map: " << ex.what());
      return;
    }
  }

  auto waypt_state = global_.setState(ROS2Eigen<float>(pose_map_frame));
  publishLocalGoal(pose_msg->header.stamp);

  if (waypt_state == WaypointManager::WaypointState::GOAL_REACHED) {
    cancelLocalPlanner();
    if (using_action_server_) {
      timeout_timer_.stop();
      spomp::GlobalNavigateResult result;
      result.status = spomp::GlobalNavigateResult::SUCCESS;
      global_navigate_as_.setSucceeded(result);
    }
  }
}

/**
* @brief Callback function for setting the goal.
*
* This function is called when a new goal message is received.
* It sets the goal based on the received message.
*
* @param goal_msg A const pointer to the received goal message of type geometry_msgs::PoseStamped.
*
* @return void
*/
    void GlobalWrapper::goalSimpleCallback(
    const geometry_msgs::PoseStamped::ConstPtr& goal_msg) 
{
  using_action_server_ = false;
  setGoal(*goal_msg);
}

/**
 * @brief Callback function for reachability data.
 *
 * This function is called when new reachability data is received. It performs the following steps:
 * - Looks up the panoramic pose at the time of the scan.
 * - Converts the reachability message to a Reachability object.
 * - Sets the pose of the Reachability object using the pose32pose2 function.
 * - Updates the local reachability map using the Reachability object.
 * - Calls globalNavigateSetFailed function if updating the reachability map fails.
 * - Visualizes the path, graph, aerial map, and reachability history.
 * - Prints the timing information.
 *
 * @param reachability_msg The received reachability data.
 */
    void GlobalWrapper::reachabilityCallback(
    const sensor_msgs::LaserScan::ConstPtr& reachability_msg) 
{
  // Get pano pose at time of scan
  geometry_msgs::TransformStamped reach_pose_msg;
  try {
    reach_pose_msg = tf_buffer_.lookupTransform(
        "map", reachability_msg->header.frame_id, reachability_msg->header.stamp, 
        ros::Duration(0.01));
  } catch (tf2::TransformException& ex) {
    ROS_ERROR_STREAM("[SPOMP-Global] Cannot get reachability pano pose: " << ex.what());
    return;
  }
  Reachability reachability = ConvertFromROS(*reachability_msg);
  reachability.setPose(pose32pose2(ROS2Eigen<float>(reach_pose_msg)));

  if (!global_.updateLocalReachability(reachability)) {
    globalNavigateSetFailed();
  }
  visualizePath(reachability_msg->header.stamp);
  visualizeGraph(reachability_msg->header.stamp);
  visualizeAerialMap(reachability_msg->header.stamp);
  publishReachabilityHistory(reachability_msg->header.stamp);
  printTimings();
}

/**
 * @brief Callback function for receiving other robot's reachability data
 *
 * This function is a callback for receiving other robot's reachability data.
 * It checks if the reachability scan is new (i.e., hasn't been seen before),
 * and if so, it converts the data to the appropriate format, updates the
 * global reachability map, and sets the pose and type of the reachability data.
 * Finally, it updates the visualization of the path and graph if any updates have occurred.
 *
 * @param robot_id The ID of the other robot
 * @param reachability_msg Pointer to the message containing reachability data
 */
    void GlobalWrapper::otherReachabilityCallback(int robot_id,
                                                  const LocalReachabilityArray::ConstPtr& reachability_msg)
{
  bool updates_occurred = false;
  for (const auto& reach : reachability_msg->reachabilities) {
    uint64_t stamp = reach.reachability.header.stamp.toNSec();
    if (!global_.haveReachabilityForRobotAtStamp(robot_id, stamp)) {
      // We haven't seen this reachability scan before
      Reachability reachability = ConvertFromROS(reach.reachability);
      reachability.setIsOtherRobot();
      reachability.setPose(ROS2Eigen<float>(reach.pose));

      if (!global_.updateOtherLocalReachability(reachability, robot_id)) {
        globalNavigateSetFailed();
      }
      updates_occurred = true;
    }
  }

  if (updates_occurred) {
    visualizePath(ros::Time::now());
    visualizeGraph(ros::Time::now());
  }
}

/**
 * @brief Executes the callback function for the timeout timer event.
 *
 * This function is called when the timeout timer event is triggered. It performs the following actions:
 *
 * 1. Cancels the global planner.
 * 2. Cancels the local planner.
 * 3. Visualizes the current path.
 * 4. If an action server is being used, stops the timeout timer and sets the global navigate result to TIMEOUT.
 *
 * @param event The timer event that triggered the callback.
 */
    void GlobalWrapper::timeoutTimerCallback(const ros::TimerEvent& event) {
  global_.cancel();
  cancelLocalPlanner();
  visualizePath(ros::Time::now());
  if (using_action_server_) {
    timeout_timer_.stop();
    spomp::GlobalNavigateResult result;
    result.status = spomp::GlobalNavigateResult::TIMEOUT;
    global_navigate_as_.setAborted(result);
  }
}

/**
* @brief Callback function for the global navigate goal action.
*
* This function is called when a new goal is received for the global navigate action server.
* It retrieves the goal from the action server and attempts to set the goal in the global planner.
* If setting the goal is not successful and the 'force' flag is set in the goal, the function
* attempts to reset the graph and set the goal again. If setting the goal is successful, a timer
* is started based on the timeout duration provided by the global planner. If the timeout duration
* is greater than 0, a timer is created using ROS timer callbacks. If setting the goal is not
* successful and the 'force' flag is not set, the function sets an appropriate result status
* indicating that a path could not be found and marks the action server as aborted.
*/
    void GlobalWrapper::globalNavigateGoalCallback() {
  using_action_server_ = true;
  auto goal = global_navigate_as_.acceptNewGoal();
  bool success = setGoal(goal->goal);

  if (!success && goal->force) {
    std::cout << "\033[31m" << "[SPOMP-Global] Attemping to reset graph" 
      << "\033[0m" << std::endl;
    global_.resetGraphLocked();
    success = setGoal(goal->goal);
  }

  if (!success) {
    // Failed to find a path
    spomp::GlobalNavigateResult result;
    result.status = spomp::GlobalNavigateResult::NO_PATH;
    global_navigate_as_.setAborted(result);
  } else {
    // Start timer
    float duration = global_.getTimeoutDuration();
    if (duration > 0) {
      timeout_timer_ = nh_.createTimer(ros::Duration(duration), 
          &GlobalWrapper::timeoutTimerCallback, this, true);
    }
  }
}

/**
 * @brief Callback function for preempting the global navigation.
 *
 * This function is called when a preempt event occurs in the global navigation action server.
 * It cancels the ongoing global navigation, cancels the local planner, visualizes the path,
 * stops the timeout timer, and sends a preempted result to the action client.
 *
 * @note This function assumes the existence of the following member variables and functions:
 *       - global_: An instance of a class that provides global navigation capabilities.
 *       - cancelLocalPlanner(): A function that cancels the local planner by sending a goal to the current pose.
 *       - visualizePath(ros::Time): A function that visualizes the current path.
 *       - timeout_timer_: A timer object used for timeout handling.
 *       - global_navigate_as_: An instance of an action server used for global navigation.
 *
 * @return None.
 */
    void GlobalWrapper::globalNavigatePreemptCallback() {
  // Stop was commanded
  global_.cancel();
  cancelLocalPlanner();
  visualizePath(ros::Time::now());
  timeout_timer_.stop();
  spomp::GlobalNavigateResult result;
  result.status = spomp::GlobalNavigateResult::CANCELLED;
  global_navigate_as_.setPreempted(result);
}

/**
 * @brief Sets the global navigation to a failed state.
 *
 * This function cancels the global navigation, cancels the local planner,
 * visualizes the current path, and if using the action server, it sets the
 * result to failed and aborts the action.
 */
    void GlobalWrapper::globalNavigateSetFailed() {
  global_.cancel();
  cancelLocalPlanner();
  visualizePath(ros::Time::now());
  if (using_action_server_) {
    timeout_timer_.stop();
    spomp::GlobalNavigateResult result;
    result.status = spomp::GlobalNavigateResult::FAILED;
    global_navigate_as_.setAborted(result);
  }
}

/**
 * @brief Sets the goal for the global planner.
 *
 * This function sets the goal for the global planner. It first resets the last goal to ensure
 * that if the same goal is provided again, it will be retransmitted. Then, it checks if the
 * provided goal is in the map frame. If not, it transforms the goal using the most recent
 * transform from the odom frame to the map frame. If the transformation fails, an error message
 * is logged and `false` is returned.
 *
 * Finally, the transformed goal position is passed to the `setGoal` function of the `global_` object.
 * The `visualizePath` and `visualizeGraph` functions are called to visualize the path and graph
 * associated with the goal. The success status of setting the goal is returned.
 *
 * @param goal_msg The goal pose stamped message containing the target position and orientation.
 * @return `true` if the goal is set successfully, otherwise `false`.
 */
// Shared callback for action and simple interfaces
bool GlobalWrapper::setGoal(
    const geometry_msgs::PoseStamped& goal_msg) 
{
  // Reset last_goal so that if we get same goal as old we retransmit it
  last_goal_.setZero();

  auto goal_map_frame = goal_msg;
  if (goal_map_frame.header.frame_id != map_frame_) {
    try {
      // Ask for most recent transform, treating the odom frame as fixed
      // This essentially tells tf to assume that the odom frame has not changed
      // since the last update, which is a valid assumption to make.
      goal_map_frame = tf_buffer_.transform(goal_msg, map_frame_, ros::Time(0), 
          goal_msg.header.frame_id);
    } catch (tf2::TransformException& ex) {
      ROS_ERROR_STREAM("Cannot transform goal to map: " << ex.what());
      return false;
    }
  }

  bool success = global_.setGoal(ROS2Eigen<float>(goal_map_frame).translation());
  visualizePath(goal_msg.header.stamp);
  visualizeGraph(goal_msg.header.stamp);
  return success;
}

/**
 * @brief Publishes the local goal if it has changed.
 *
 * This function publishes the local goal if there is a new goal compared to the previous one.
 * It creates a geometry_msgs::PoseStamped message with the local goal and publishes it on the local_goal_pub_ topic.
 * The local goal message contains the position of the goal in the map frame.
 *
 * @param stamp The timestamp for the local goal message.
 */
    void GlobalWrapper::publishLocalGoal(const ros::Time& stamp) {
  auto cur_goal = global_.getNextWaypoint();
  if (!cur_goal) return;

  if ((last_goal_ - *cur_goal).norm() > 0.0001) {
    // We have a new goal, publish it
    geometry_msgs::PoseStamped local_goal_msg;
    local_goal_msg.header.stamp = stamp;
    local_goal_msg.header.frame_id = map_frame_;
    local_goal_msg.pose.position.x = (*cur_goal)[0];
    local_goal_msg.pose.position.y = (*cur_goal)[1];
    local_goal_msg.pose.orientation.w = 1;
    local_goal_pub_.publish(local_goal_msg);

    last_goal_ = *cur_goal;
  }
}

/**
 * @brief Cancels the local planner by sending a goal of the current pose.
 *
 * The last_goal_ is reset so that if the same goal is received again, it will be retransmitted.
 * The current position is obtained from the GlobalPlanner object. If the position is not available, the function returns early.
 * A geometry_msgs::PoseStamped message is created and populated with the current pose information.
 * The message header is set with the current timestamp and the map_frame_ as the frame_id.
 * The local goal message is published to the local_goal_pub_ topic.
 */
    void GlobalWrapper::cancelLocalPlanner() {
  // Reset last_goal so that if we get same goal as old we retransmit it
  last_goal_.setZero();

  auto pos = global_.getPos();
  if (!pos) return;

  // Send goal of current pose to effectively cancel
  geometry_msgs::PoseStamped local_goal_msg;
  local_goal_msg.header.stamp = ros::Time::now();
  local_goal_msg.header.frame_id = map_frame_;
  local_goal_msg.pose.position.x = (*pos)[0];
  local_goal_msg.pose.position.y = (*pos)[1];
  local_goal_msg.pose.orientation.w = 1;
  local_goal_pub_.publish(local_goal_msg);
}

/**
 * @brief Publishes the reachability history.
 *
 * This function publishes the reachability history if it is available.
 * It creates a LocalReachabilityArray message and fills it with the reachability data.
 * The message is then published using the reachability_history_pub_ publisher.
 *
 * @param stamp The timestamp to be set in the message header.
 */
    void GlobalWrapper::publishReachabilityHistory(const ros::Time& stamp) {
  if (!global_.haveNewReachabilityHistory()) {
    return;
  }

  const auto& reach_hist = global_.getReachabilityHistory();

  LocalReachabilityArray lra_msg;
  lra_msg.header.stamp = stamp;
  lra_msg.header.frame_id = map_frame_;
  lra_msg.reachabilities.reserve(reach_hist.size());
  for (const auto& [stamp, reach] : reach_hist) {
    LocalReachability lr_msg; 
    lr_msg.reachability = Convert2ROS(reach);
    lr_msg.pose = Eigen2ROS2D(reach.getPose());
    lra_msg.reachabilities.push_back(lr_msg);
  }

  reachability_history_pub_.publish(lra_msg);
}

/**
 * Visualizes the graph by publishing visualization markers and map images.
 *
 * @param stamp The timestamp for the visualization.
 */
    void GlobalWrapper::visualizeGraph(const ros::Time& stamp) {
  visualization_msgs::MarkerArray viz_msg;

  visualization_msgs::Marker edge_viz;
  edge_viz.header.stamp = stamp;
  edge_viz.header.frame_id = map_frame_;
  edge_viz.ns = "edge_viz";
  edge_viz.id = 0;
  edge_viz.type = visualization_msgs::Marker::LINE_LIST;
  edge_viz.action = visualization_msgs::Marker::ADD;
  edge_viz.pose.orientation.w = 1;
  edge_viz.scale.x = 0.2; // Width of segment
  edge_viz.color.a = 1;

  for (const auto& edge : global_.getEdges()) {
    Eigen::Vector3f point_3 = {0, 0, 1};
    point_3.head<2>() = edge.node1->pos;
    geometry_msgs::Point pt_msg = Eigen2ROS(point_3);
    edge_viz.points.push_back(pt_msg);

    point_3.head<2>() = edge.node2->pos;
    pt_msg = Eigen2ROS(point_3);
    edge_viz.points.push_back(pt_msg);

    std_msgs::ColorRGBA color;
    color.a = 1;
    if (edge.is_locked) {
      if (edge.cls == 0) {
        color.g = 1;
      } else {
        color.r = 1;
      }
    } else {
      float color_mag = std::min<float>(1./(edge.cost*edge.length), 1);
      float hue = (static_cast<float>(edge.cls)/
          (TravGraph::Edge::MAX_TERRAIN-1))*(1./2) + 1./2;
      Eigen::Vector3f rgb = hsv2rgb({hue, 0.5, color_mag});
      color.r = rgb[0];
      color.g = rgb[1];
      color.b = rgb[2];
    }

    // Do twice to handle both endpts
    edge_viz.colors.push_back(color);
    edge_viz.colors.push_back(color);
  }

  viz_msg.markers.push_back(edge_viz);
  graph_viz_pub_.publish(viz_msg);

  // Publish as image as well
  sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(edge_viz.header, "bgr8", 
      global_.getMapImageViz()).toImageMsg();
  map_img_viz_pub_.publish(img_msg);
}

/**
 * @brief Visualizes a path using ROS
 *
 * This method publishes a path visualization using the specified stamp. The path
 * is obtained from the `global_` object and converted into a `nav_msgs::Path` message.
 * The path is visualized by publishing the message to the `path_viz_pub_` topic.
 *
 * @param stamp The timestamp to be used in the header of the `nav_msgs::Path` message
 */
    void GlobalWrapper::visualizePath(const ros::Time& stamp) {
  nav_msgs::Path path_msg;
  path_msg.header.stamp = stamp;
  path_msg.header.frame_id = map_frame_;

  geometry_msgs::PoseStamped path_pose_msg;
  path_pose_msg.header = path_msg.header;
  path_pose_msg.pose.orientation.w = 1;
  path_pose_msg.pose.position.z = 1;
  for (const auto& node : global_.getPath()) {
    path_pose_msg.pose.position.x = node->pos[0];
    path_pose_msg.pose.position.y = node->pos[1];

    path_msg.poses.push_back(path_pose_msg);
  }

  path_viz_pub_.publish(path_msg);
}

/**
 * @brief Visualize the aerial map
 *
 * This function takes in a timestamp and publishes the aerial map as an image message.
 * The aerial map is obtained from the global object and converted to an image message using CvBridge.
 * The converted image message is then published using the aerial_map_trav_viz_pub_ ROS publisher.
 *
 * @param stamp The timestamp for the image message header
 */
    void GlobalWrapper::visualizeAerialMap(const ros::Time& stamp) {
  std_msgs::Header header;
  header.stamp = stamp;
  header.frame_id = map_frame_;
  sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(header, "bgr8", 
      global_.getAerialMapTrav()).toImageMsg();
  aerial_map_trav_viz_pub_.publish(img_msg);
}

void GlobalWrapper::printTimings() {
  ROS_INFO_STREAM("\033[34m" << "[SPOMP-Global]" << std::endl << 
      TimerManager::getGlobal() << "\033[0m");
}

} // namespace spomp
