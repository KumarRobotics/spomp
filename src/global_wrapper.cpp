#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <cv_bridge/cv_bridge.h>
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
      ROS_INFO_STREAM("Got new map");
      last_map_stamp_ = loc_it->first;

      global_.updateMap(
          cv_bridge::toCvShare(img_it->second, sensor_msgs::image_encodings::BGR8)->image,
          Eigen::Vector2f(loc_it->second->point.x, loc_it->second->point.y));

      //Clean up buffers
      map_sem_buf_.erase(map_sem_buf_.begin(), ++img_it);
      map_loc_buf_.erase(map_loc_buf_.begin(), loc_it.base());
      break;
    }
  }
}

void GlobalWrapper::goalCallback(
    const geometry_msgs::PoseStamped::ConstPtr& goal_msg) 
{
}

void GlobalWrapper::printTimings() {
  ROS_INFO_STREAM("\033[34m" << TimerManager::getGlobal() << "\033[0m");
}

} // namespace spomp
