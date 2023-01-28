#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "spomp/global_wrapper.h"
#include "spomp/timer.h"
#include "spomp/rosutil.h"

namespace spomp {

std::string GlobalWrapper::odom_frame_{"odom"};
std::string GlobalWrapper::map_frame_{"map"};
std::string GlobalWrapper::robot_list_{"robot"};
std::string GlobalWrapper::this_robot_{"robot"};

GlobalWrapper::GlobalWrapper(ros::NodeHandle& nh) : 
  nh_(nh), 
  global_(createGlobal(nh)),
  tf_buffer_(),
  tf_listener_(tf_buffer_),
  global_navigate_as_(nh_, "navigate", false)
{
  // Publishers
  // Latch goal so it is received even if local planner hasn't started yet
  local_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("local_goal", 1, true);
  reachability_history_pub_ = nh_.advertise<LocalReachabilityArray>("reachability_history", 1, true);
  graph_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("graph_viz", 1, true);
  path_viz_pub_ = nh_.advertise<nav_msgs::Path>("path_viz", 1, true);
  map_img_viz_pub_ = nh_.advertise<sensor_msgs::Image>("viz_img", 1);
  aerial_map_trav_viz_pub_ = nh_.advertise<sensor_msgs::Image>("aerial_map_trav_viz", 1);
}

Global GlobalWrapper::createGlobal(ros::NodeHandle& nh) {
  TravMap::Params tm_params{};
  TravGraph::Params tg_params{};
  AerialMap::Params am_params{};
  MLPModel::Params mlp_params{};
  WaypointManager::Params wm_params{};

  nh.getParam("world_config_path", tm_params.world_config_path);
  nh.getParam("robot_list", robot_list_);
  nh.getParam("this_robot", this_robot_);

  nh.getParam("TM_max_hole_fill_size_m", tm_params.max_hole_fill_size_m);
  nh.getParam("TM_vis_dist_m", tm_params.vis_dist_m);
  nh.getParam("TM_unvis_start_thresh", tm_params.unvis_start_thresh);
  nh.getParam("TM_unvis_stop_thresh", tm_params.unvis_stop_thresh);
  nh.getParam("TM_prune", tm_params.prune);

  nh.getParam("TG_reach_node_max_dist_m", tg_params.reach_node_max_dist_m);
  nh.getParam("TG_trav_window_rad", tg_params.trav_window_rad);
  nh.getParam("TG_max_trav_discontinuity_m", tg_params.max_trav_discontinuity_m);
  nh.getParam("TG_num_untrav_before_mark", tg_params.num_untrav_before_mark);

  nh.getParam("AM_trav_thresh", am_params.trav_thresh);
  nh.getParam("AM_not_trav_thresh", am_params.not_trav_thresh);
  nh.getParam("AM_not_trav_range_m", am_params.not_trav_range_m);

  nh.getParam("MLP_hidden_layer_size", mlp_params.hidden_layer_size);
  nh.getParam("MLP_regularization", mlp_params.regularization);

  nh.getParam("WM_waypoint_thresh_m", wm_params.waypoint_thresh_m);

  constexpr int width = 30;
  using namespace std;
  ROS_INFO_STREAM("\033[32m" << "[SPOMP-Global]" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(width) << "[ROS] world_config_path: " << tm_params.world_config_path << endl <<
    setw(width) << "[ROS] robot_list: " << robot_list_ << endl <<
    setw(width) << "[ROS] this_robot: " << this_robot_ << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] TM_max_hole_fill_size_m: " << tm_params.max_hole_fill_size_m << endl <<
    setw(width) << "[ROS] TM_vis_dist_m: " << tm_params.vis_dist_m << endl <<
    setw(width) << "[ROS] TM_unvis_start_thresh: " << tm_params.unvis_start_thresh << endl <<
    setw(width) << "[ROS] TM_unvis_stop_thresh: " << tm_params.unvis_stop_thresh << endl <<
    setw(width) << "[ROS] TM_prune: " << tm_params.prune << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] AM_trav_thresh: " << am_params.trav_thresh << endl <<
    setw(width) << "[ROS] AM_not_trav_thresh: " << am_params.not_trav_thresh << endl <<
    setw(width) << "[ROS] AM_not_trav_range_m: " << am_params.not_trav_range_m << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] MLP_hidden_layer_size: " << mlp_params.hidden_layer_size << endl <<
    setw(width) << "[ROS] MLP_regularization: " << mlp_params.regularization << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] TG_reach_node_max_dist_m: " << tg_params.reach_node_max_dist_m << endl <<
    setw(width) << "[ROS] TG_trav_window_rad: " << tg_params.trav_window_rad << endl <<
    setw(width) << "[ROS] TG_max_trav_discontinuity_m: " << tg_params.max_trav_discontinuity_m << endl <<
    setw(width) << "[ROS] TG_num_untrav_before_mark: " << tg_params.num_untrav_before_mark << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] WM_waypoint_thresh_m: " << wm_params.waypoint_thresh_m << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");

  return Global(tm_params, tg_params, am_params, mlp_params, wm_params);
}

void GlobalWrapper::initialize() {
  // Subscribers
  map_sem_img_sub_ = nh_.subscribe("map_sem_img", 1, &GlobalWrapper::mapSemImgCallback, this);
  map_sem_img_center_sub_ = nh_.subscribe("map_sem_img_center", 1, 
      &GlobalWrapper::mapSemImgCenterCallback, this);
  pose_sub_ = nh_.subscribe("pose", 1, &GlobalWrapper::poseCallback, this);
  goal_sub_ = nh_.subscribe("goal_simple", 1, &GlobalWrapper::goalSimpleCallback, this);
  reachability_sub_ = nh_.subscribe("reachability", 1, 
      &GlobalWrapper::reachabilityCallback, this);

  std::istringstream robots(robot_list_);
  std::string robot;
  int id = 0;
  while (std::getline(robots, robot, ',')) {
    if (robot == this_robot_) continue;
    other_robot_reachability_subs_.push_back(nh_.subscribe<LocalReachabilityArray>(
          "/" + robot + "/spomp_global/reachability_history", 1, 
          std::bind(&GlobalWrapper::otherReachabilityCallback, this, id, std::placeholders::_1)));
    ++id;
  }
  other_reachability_list_.resize(id);

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

void GlobalWrapper::mapSemImgCallback(
    const sensor_msgs::Image::ConstPtr& img_msg) 
{
  if (img_msg->header.stamp.toNSec() > last_map_stamp_) {
    // Initial sanity check
    if (img_msg->height > 0 && img_msg->width > 0) {
      ROS_DEBUG("Got map img");
      map_sem_buf_.insert({img_msg->header.stamp.toNSec(), img_msg});
      processMapBuffers();
    }
  }
}

void GlobalWrapper::mapSemImgCenterCallback(
    const geometry_msgs::PointStamped::ConstPtr& pt_msg) 
{
  if (pt_msg->header.stamp.toNSec() > last_map_stamp_) {
    ROS_DEBUG("Got map loc");
    map_loc_buf_.insert({pt_msg->header.stamp.toNSec(), pt_msg});
    processMapBuffers();
  }
}

void GlobalWrapper::processMapBuffers() {
  // Loop starting with most recent
  for (auto loc_it = map_loc_buf_.rbegin(); loc_it != map_loc_buf_.rend(); ++loc_it) {
    auto img_it = map_sem_buf_.find(loc_it->first);
    if (img_it != map_sem_buf_.end()) {
      ROS_DEBUG_STREAM("Got new map");
      last_map_stamp_ = loc_it->first;

      global_.updateMap(
          cv_bridge::toCvCopy(img_it->second, sensor_msgs::image_encodings::MONO8)->image,
          {loc_it->second->point.x, loc_it->second->point.y});
      visualizeGraph(loc_it->second->header.stamp);

      //Clean up buffers
      map_sem_buf_.erase(map_sem_buf_.begin(), ++img_it);
      map_loc_buf_.erase(map_loc_buf_.begin(), loc_it.base());
      break;
    }
  }
}

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

  bool path_complete = global_.setState(ROS2Eigen<float>(pose_map_frame));
  publishLocalGoal(pose_msg->header.stamp);

  if (path_complete && using_action_server_) {
    spomp::GlobalNavigateResult result;
    result.status = spomp::GlobalNavigateResult::SUCCESS;
    global_navigate_as_.setSucceeded(result);
  }
}

void GlobalWrapper::goalSimpleCallback(
    const geometry_msgs::PoseStamped::ConstPtr& goal_msg) 
{
  using_action_server_ = false;
  setGoal(*goal_msg);
}

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
    ROS_ERROR_STREAM("Cannot get reachability pano pose: " << ex.what());
    return;
  }
  Reachability reachability = ConvertFromROS(*reachability_msg);
  reachability.setPose(pose32pose2(ROS2Eigen<float>(reach_pose_msg)));

  global_.updateLocalReachability(reachability);
  visualizePath(reachability_msg->header.stamp);
  visualizeGraph(reachability_msg->header.stamp);
  visualizeAerialMap(reachability_msg->header.stamp);
  publishReachabilityHistory();
  printTimings();
}

void GlobalWrapper::otherReachabilityCallback(int robot_id,
    const LocalReachabilityArray::ConstPtr& reachability_msg) 
{
  bool updates_occurred = false;
  for (const auto& reach : reachability_msg->reachabilities) {
    uint64_t stamp = reach.reachability.header.stamp.toNSec();
    if (other_reachability_list_[robot_id].count(stamp) == 0) {
      // We haven't seen this reachability scan before
      Reachability reachability = ConvertFromROS(reach.reachability);
      reachability.setPose(ROS2Eigen<float>(reach.pose));
      global_.updateOtherLocalReachability(reachability);
      other_reachability_list_[robot_id].insert(stamp);
      updates_occurred = true;
    }
  }

  if (updates_occurred) {
    visualizePath(ros::Time::now());
    visualizeGraph(ros::Time::now());
  }
}

void GlobalWrapper::globalNavigateGoalCallback() {
  using_action_server_ = true;
  auto goal = global_navigate_as_.acceptNewGoal();
  bool success = setGoal(goal->goal);

  if (!success) {
    // Failed to find a path
    spomp::GlobalNavigateResult result;
    result.status = spomp::GlobalNavigateResult::NO_PATH;
    global_navigate_as_.setAborted(result);
  }
}

void GlobalWrapper::globalNavigatePreemptCallback() {
  // Stop was commanded
  global_.cancel();
  spomp::GlobalNavigateResult result;
  result.status = spomp::GlobalNavigateResult::CANCELLED;
  global_navigate_as_.setPreempted(result);
}

// Shared callback for action and simple interfaces
bool GlobalWrapper::setGoal(
    const geometry_msgs::PoseStamped& goal_msg) 
{
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

void GlobalWrapper::publishReachabilityHistory() {
  const auto& reach_hist = global_.getReachabilityHistory();

  LocalReachabilityArray lra_msg;
  lra_msg.reachabilities.reserve(reach_hist.size());
  for (const auto& reachability : reach_hist) {
    LocalReachability lr_msg; 
    lr_msg.reachability = Convert2ROS(reachability);
    lr_msg.pose = Eigen2ROS2D(reachability.getPose());
    lra_msg.reachabilities.push_back(lr_msg);
  }

  reachability_history_pub_.publish(lra_msg);
}

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
    float color_mag = std::min<float>(1./edge.cost, 1);
    if (edge.cls == 0) {
      if (edge.is_experienced) {
        color.g = color_mag;
      } else {
        color.g = color_mag;
        color.r = color_mag;
      }
    } else if (edge.cls == 1) {
      color.g = color_mag;
      color.b = color_mag;
    } else if (edge.cls == 2) {
      color.b = color_mag;
    } else if (edge.cls == 3) {
      color.r = color_mag;
      color.b = color_mag;
    }

    if (edge.is_experienced && edge.cls > 0) {
      // Known bad
      color.r = color_mag;
    }
    // Otherwise leave black

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
