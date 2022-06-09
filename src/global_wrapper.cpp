#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include "spomp/global_wrapper.h"
#include "spomp/timer.h"

namespace spomp {

GlobalWrapper::GlobalWrapper(ros::NodeHandle& nh) : 
  nh_(nh), 
  global_(createGlobal(nh))
{
  // Publishers
  local_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("local_goal", 1);
  graph_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("graph_viz", 1);
  path_viz_pub_ = nh_.advertise<nav_msgs::Path>("path_viz", 1);
}

Global GlobalWrapper::createGlobal(ros::NodeHandle& nh) {
  TravMap::Params tm_params{};
  WaypointManager::Params wm_params{};

  nh.getParam("TM_terrain_types_path", tm_params.terrain_types_path);
  nh.getParam("TM_map_res", tm_params.map_res);
  nh.getParam("TM_max_hole_fill_size_m", tm_params.max_hole_fill_size_m);
  nh.getParam("TM_vis_dist_m", tm_params.vis_dist_m);

  nh.getParam("WM_waypoint_thresh_m", wm_params.waypoint_thresh_m);

  constexpr int width = 30;
  using namespace std;
  ROS_INFO_STREAM("\033[32m" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(width) << "[ROS] TM_terrain_types_path: " << tm_params.terrain_types_path << endl <<
    setw(width) << "[ROS] TM_map_res: " << tm_params.map_res << endl <<
    setw(width) << "[ROS] TM_max_hole_fill_size_m: " << tm_params.max_hole_fill_size_m << endl <<
    setw(width) << "[ROS] TM_vis_dist_m: " << tm_params.vis_dist_m << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] WM_waypoint_thresh_m: " << wm_params.waypoint_thresh_m << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");

  return Global(tm_params, wm_params);
}

void GlobalWrapper::initialize() {
  // Subscribers
  map_sem_img_sub_ = nh_.subscribe("map_sem_img", 1, &GlobalWrapper::mapSemImgCallback, this);
  map_sem_img_center_sub_ = nh_.subscribe("map_sem_img_center", 1, 
      &GlobalWrapper::mapSemImgCenterCallback, this);
  goal_sub_ = nh_.subscribe("goal", 1, &GlobalWrapper::goalCallback, this);

  ros::spin();
}

void GlobalWrapper::mapSemImgCallback(
    const sensor_msgs::Image::ConstPtr& img_msg) 
{
}

void GlobalWrapper::mapSemImgCenterCallback(
    const geometry_msgs::PointStamped::ConstPtr& pt_msg) 
{
}

void GlobalWrapper::goalCallback(
    const geometry_msgs::PoseStamped::ConstPtr& goal_msg) 
{
}

void GlobalWrapper::printTimings() {
  ROS_INFO_STREAM("\033[34m" << TimerManager::getGlobal() << "\033[0m");
}

} // namespace spomp
