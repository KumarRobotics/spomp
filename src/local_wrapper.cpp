#include "spomp/local_wrapper.h"
#include "spomp/rosutil.h"
#include <iomanip>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

namespace spomp {

std::string LocalWrapper::odom_frame_{"odom"};
std::string LocalWrapper::pano_frame_{"planner_pano"};
std::string LocalWrapper::body_frame_{"body"};
std::string LocalWrapper::control_frame_{"base_link"};

LocalWrapper::LocalWrapper(ros::NodeHandle& nh) : 
  nh_(nh), 
  local_(createLocal(nh)), 
  remote_(50), 
  it_(nh),
  tf_buffer_(),
  tf_listener_(tf_buffer_),
  tf_broadcaster_()
{
  auto& tm = TimerManager::getGlobal();
  viz_pano_t_ = tm.get("LW_viz_pano");
  viz_cloud_t_ = tm.get("LW_viz_cloud");

  // Publishers
  // Viz
  obs_pano_viz_pub_ = it_.advertise("obs_pano_viz", 1);
  obs_cloud_viz_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obs_cloud_viz", 1);
  control_viz_pub_ = nh_.advertise<nav_msgs::Path>("control_viz", 1);
  local_goal_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("local_goal_viz", 1);
  // Control
  control_pub_ = nh_.advertise<geometry_msgs::Twist>("control", 1);
  // Publish reachability for higher level planner
  reachability_pub_ = nh_.advertise<sensor_msgs::LaserScan>("reachability", 1);
}

Local LocalWrapper::createLocal(ros::NodeHandle& nh) {
  Local::Params sl_params{};
  TerrainPano::Params tp_params{};
  PanoPlanner::Params pp_params{};
  Controller::Params co_params{};

  nh.getParam("tbb", tp_params.tbb);
  nh.getParam("tbb", pp_params.tbb);
  
  nh.getParam("odom_frame", odom_frame_);
  nh.getParam("planner_pano_frame", pano_frame_);
  nh.getParam("body_frame", body_frame_);
  nh.getParam("control_frame", control_frame_);

  nh.getParam("SL_global_goal_thresh_m", sl_params.global_goal_thresh_m);
  nh.getParam("SL_local_goal_thresh_m", sl_params.local_goal_thresh_m);

  nh.getParam("TP_max_hole_fill_size", tp_params.max_hole_fill_size);
  nh.getParam("TP_min_noise_size", tp_params.min_noise_size);
  nh.getParam("TP_v_fov_rad", tp_params.v_fov_rad);
  nh.getParam("TP_target_dist_xy", tp_params.target_dist_xy);
  nh.getParam("TP_noise_m", tp_params.noise_m);
  nh.getParam("TP_slope_thresh", tp_params.slope_thresh);
  nh.getParam("TP_inflation_m", tp_params.inflation_m);
  nh.getParam("TP_max_distance_m", tp_params.max_distance_m);

  nh.getParam("PP_max_spacing_m", pp_params.max_spacing_m);
  nh.getParam("PP_sample_size", pp_params.sample_size);
  nh.getParam("PP_consistency_cost", pp_params.consistency_cost);

  nh.getParam("CO_freq", co_params.freq);
  nh.getParam("CO_max_lin_accel", co_params.max_lin_accel);
  nh.getParam("CO_max_ang_accel", co_params.max_ang_accel);
  nh.getParam("CO_max_lin_vel", co_params.max_lin_vel);
  nh.getParam("CO_max_ang_vel", co_params.max_ang_vel);
  nh.getParam("CO_horizon_sec", co_params.horizon_sec);
  nh.getParam("CO_horizon_dt", co_params.horizon_dt);
  nh.getParam("CO_lin_disc", co_params.lin_disc);
  nh.getParam("CO_ang_disc", co_params.ang_disc);
  nh.getParam("CO_obs_cost_weight", co_params.obs_cost_weight);

  bool have_trans = getControlTrans(co_params.control_trans);

  constexpr int width = 30;
  using namespace std;
  ROS_INFO_STREAM("\033[32m" << "[SPOMP-Local]" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(width) << "[ROS] tbb: " << tp_params.tbb << endl <<
    setw(width) << "[ROS] odom_frame: " << odom_frame_ << endl <<
    setw(width) << "[ROS] planner_pano_frame: " << pano_frame_ << endl <<
    setw(width) << "[ROS] body_frame: " << body_frame_ << endl <<
    setw(width) << "[ROS] control_frame: " << control_frame_ << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] SL_global_goal_thresh_m: " << sl_params.global_goal_thresh_m << endl <<
    setw(width) << "[ROS] SL_local_goal_thresh_m: " << sl_params.local_goal_thresh_m << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] TP_max_hole_fill_size: " << tp_params.max_hole_fill_size << endl <<
    setw(width) << "[ROS] TP_min_noise_size: " << tp_params.min_noise_size << endl <<
    setw(width) << "[ROS] TP_v_fov_rad: " << tp_params.v_fov_rad << endl <<
    setw(width) << "[ROS] TP_target_dist_xy: " << tp_params.target_dist_xy << endl <<
    setw(width) << "[ROS] TP_noise_m: " << tp_params.noise_m << endl <<
    setw(width) << "[ROS] TP_slope_thresh: " << tp_params.slope_thresh << endl <<
    setw(width) << "[ROS] TP_inflation_m: " << tp_params.inflation_m << endl <<
    setw(width) << "[ROS] TP_max_distance_m: " << tp_params.max_distance_m << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] PP_max_spacing_m: " << pp_params.max_spacing_m << endl <<
    setw(width) << "[ROS] PP_sample_size: " << pp_params.sample_size << endl <<
    setw(width) << "[ROS] PP_consistency_cost: " << pp_params.consistency_cost << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] CO_freq: " << co_params.freq << endl <<
    setw(width) << "[ROS] CO_max_lin_accel: " << co_params.max_lin_accel << endl <<
    setw(width) << "[ROS] CO_max_ang_accel: " << co_params.max_ang_accel << endl <<
    setw(width) << "[ROS] CO_max_lin_vel: " << co_params.max_lin_vel << endl <<
    setw(width) << "[ROS] CO_max_ang_vel: " << co_params.max_ang_vel << endl <<
    setw(width) << "[ROS] CO_horizon_sec: " << co_params.horizon_sec << endl <<
    setw(width) << "[ROS] CO_horizon_dt: " << co_params.horizon_dt << endl <<
    setw(width) << "[ROS] CO_lin_disc: " << co_params.lin_disc << endl <<
    setw(width) << "[ROS] CO_ang_disc: " << co_params.ang_disc << endl <<
    setw(width) << "[ROS] CO_obs_cost_weight: " << co_params.obs_cost_weight << endl <<
    setw(width) << "[ROS] CO_control_trans: " << (have_trans ? "found" : "default") << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");

  return Local(sl_params, tp_params, pp_params, co_params);
}

bool LocalWrapper::getControlTrans(Eigen::Isometry3f& trans) {
  // Static function, so create static buffer and listener
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  int retries = 0;
  while (retries < 2) {
    ros::Duration(0.5).sleep();
    try {
      auto trans_msg = tf_buffer.lookupTransform(body_frame_, control_frame_, ros::Time(0));
      trans = ROS2Eigen<float>(trans_msg);
      return true;
    } catch (tf2::TransformException& ex) {
      ROS_WARN("%s", ex.what());
      ++retries;
    }
  }

  return false;
}

void LocalWrapper::play() {
  std::string bag_path;
  if (!nh_.getParam("bag_path", bag_path)) {
    ROS_ERROR_STREAM("ERROR: No bag specified");
    return;
  }

  rosbag::Bag bag; 
  bag.open(bag_path, rosbag::bagmode::Read);

  for (const rosbag::MessageInstance& m : rosbag::View(bag)) {
    if (!ros::ok()) {
      break;
    }
    if (m.getTopic() == "/os_node/rofl_odom/pano/img") {
      sensor_msgs::Image::ConstPtr pano = m.instantiate<sensor_msgs::Image>();
      if (pano != NULL) {
        if (!remote_.wait()) {
          // Want to quit
          break;
        }

        panoCallback(pano, {});
      }
    }
    ros::spinOnce();
  }
}

void LocalWrapper::initialize() {
  // Subscribers
  pano_sub_ = it_.subscribeCamera("pano/img", 1, &LocalWrapper::panoCallback, this);
  goal_sub_ = nh_.subscribe("goal", 5, &LocalWrapper::goalCallback, this);
  pose_sub_ = nh_.subscribe("pose", 1, &LocalWrapper::poseCallback, this);

  ros::spin();
}

void LocalWrapper::panoCallback(const sensor_msgs::Image::ConstPtr& img_msg,
    const sensor_msgs::CameraInfo::ConstPtr& info_msg) 
{
  Eigen::MatrixXf pano_eig;
  cv_bridge::CvImageConstPtr pano_cv = cv_bridge::toCvShare(img_msg);
  cv::Mat depth_pano_cv(pano_cv->image.rows, pano_cv->image.cols, CV_16UC1);
  // Extract first channel (depth)
  cv::extractChannel(pano_cv->image, depth_pano_cv, 0);
  // Convert to Eigen
  cv::cv2eigen(depth_pano_cv, pano_eig);

  // Defaults
  Eigen::Isometry3f pano_pose = Eigen::Isometry3f::Identity();
  float depth_scale = 512;
  // Update using info message
  if (info_msg) {
    const auto trans_mat = 
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(&info_msg->P[0]);
    pano_pose.affine() = trans_mat.cast<float>();
    depth_scale = info_msg->R[0];
  }
  pano_eig /= depth_scale;

  Eigen::Vector3f goal = Eigen::Vector3f::Zero();
  if (last_goal_msg_.header.frame_id != "") {
    try {
      auto pose_odom_frame = tf_buffer_.transform(last_goal_msg_, odom_frame_, 
          img_msg->header.stamp, last_goal_msg_.header.frame_id, ros::Duration(0.05));
      goal = ROS2Eigen<float>(pose_odom_frame).translation();
    } catch (tf2::TransformException& ex) {
      ROS_WARN_STREAM("Unable to update goal transform: " << ex.what());
    }
  }
  local_.updatePano(pano_eig, pano_pose, goal);

  publishTransform(img_msg->header.stamp);
  publishReachability(img_msg->header.stamp);
  visualizePano(img_msg->header.stamp);
  visualizeCloud(img_msg->header.stamp);
  visualizeGoals(img_msg->header.stamp);
  printTimings();
}

void LocalWrapper::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
  try {
    // Ask for most recent transform, treating the pano frame as fixed
    // This essentially tells tf to assume that the pano frame has not changed
    // since the last update, which is a valid assumption to make.
    auto pose_odom_frame = tf_buffer_.transform(*pose_msg, odom_frame_, ros::Time(0), 
        pose_msg->header.frame_id);
    local_.setGoal(ROS2Eigen<float>(pose_odom_frame).translation());
    // Publish transform so goals are updated properly
    publishTransform(pose_msg->header.stamp);
    visualizeGoals(pose_msg->header.stamp);
    last_goal_msg_ = *pose_msg;
  } catch (tf2::TransformException& ex) {
    ROS_ERROR_STREAM("Cannot transform goal: " << ex.what());
  }
}

void LocalWrapper::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
  // Assume the pose_msg is already in the odom frame
  auto twist = local_.getControlInput(ROS2Eigen<float>(*pose_msg));
  control_pub_.publish(Eigen2ROS(twist));
  visualizeControl(pose_msg->header.stamp, twist);
}

void LocalWrapper::publishTransform(const ros::Time& stamp) {
  geometry_msgs::TransformStamped msg = Eigen2ROS(local_.getPano().getPose());
  msg.header.stamp = stamp;
  msg.header.frame_id = odom_frame_;
  msg.child_frame_id = pano_frame_;
  tf_broadcaster_.sendTransform(msg); 
}

void LocalWrapper::publishReachability(const ros::Time& stamp) {
  const auto& reachability = local_.getPlanner().getReachability();
  sensor_msgs::LaserScan scan_msg = Convert2ROS(reachability);

  scan_msg.header.frame_id = pano_frame_;
  scan_msg.header.stamp = stamp;

  reachability_pub_.publish(scan_msg);
}

void LocalWrapper::visualizePano(const ros::Time& stamp) {
  viz_pano_t_->start();

  const Eigen::MatrixXf& pano = local_.getPano().getTraversability().matrix();
  cv::Mat pano_viz;
  cv::eigen2cv(pano, pano_viz);
  pano_viz.convertTo(pano_viz, CV_8UC1, 100);

  std_msgs::Header header;
  header.stamp = stamp;
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", pano_viz).toImageMsg();
  obs_pano_viz_pub_.publish(msg);

  viz_pano_t_->end();
}

void LocalWrapper::visualizeCloud(const ros::Time& stamp) {
  viz_cloud_t_->start();

  // Grab data
  const auto& cloud = local_.getPano().getCloud();
  const auto& trav = local_.getPano().getTraversability();

  sensor_msgs::PointCloud2 cloud_msg;
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = pano_frame_;
  cloud_msg.height = 1;
  cloud_msg.width = cloud[0].size();

  {
    sensor_msgs::PointField field;
    field.name = "x";
    field.offset = 0;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }
  {
    sensor_msgs::PointField field;
    field.name = "y";
    field.offset = 4;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }
  {
    sensor_msgs::PointField field;
    field.name = "z";
    field.offset = 8;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }
  {
    sensor_msgs::PointField field;
    field.name = "data";
    field.offset = 12;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }

  cloud_msg.point_step = 16;
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
  cloud_msg.is_dense = true;

  cloud_msg.data.resize(cloud_msg.row_step*cloud_msg.height);

  // Map eigen array onto pc data
  Eigen::Map<Eigen::ArrayXXf> cloud_packed(reinterpret_cast<float*>(cloud_msg.data.data()), 
      4, cloud[0].size());

  // Copy data over
  for (int axis=0; axis<3; ++axis) {
    cloud_packed.row(axis) = Eigen::Map<const Eigen::VectorXf>(
        cloud[axis].data(), cloud[axis].size());
  }
  cloud_packed.row(3) = Eigen::Map<const Eigen::VectorXf>(
      trav.data(), trav.size());

  obs_cloud_viz_pub_.publish(cloud_msg);

  viz_cloud_t_->end();
}

void LocalWrapper::visualizeControl(const ros::Time& stamp, const Twistf& twist) {
  auto path = local_.getController().forwardFromOrigin(twist);

  nav_msgs::Path path_msg{};
  path_msg.header.frame_id = control_frame_;
  path_msg.header.stamp = stamp;
  path_msg.poses.reserve(path.size());

  for (const auto& pose : path) {
    geometry_msgs::PoseStamped pose_msg = Eigen2ROS(pose);
    pose_msg.header = path_msg.header;
    path_msg.poses.push_back(pose_msg);
  } 

  control_viz_pub_.publish(path_msg);
}

void LocalWrapper::visualizeGoals(const ros::Time& stamp) {
  auto global_goal = local_.getGlobalGoal();
  Eigen::Vector2f local_goal = local_.getController().getLocalGoal();

  visualization_msgs::MarkerArray goal_viz{};
  if (global_goal) {
    {
      visualization_msgs::Marker marker{};
      marker.header.stamp = stamp;
      marker.header.frame_id = odom_frame_;
      marker.ns = "global_goal";
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.scale.x = marker.scale.y = marker.scale.z = 3;
      marker.pose.orientation.w = 1;
      marker.color.a = 1;
      marker.color.r = 1;
      marker.pose.position.x = (*global_goal)[0];
      marker.pose.position.y = (*global_goal)[1];
      marker.pose.position.z = (*global_goal)[2];
      goal_viz.markers.push_back(marker);
    }
    {
      visualization_msgs::Marker marker{};
      marker.header.stamp = stamp;
      marker.header.frame_id = pano_frame_;
      marker.ns = "local_goal";
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.scale.x = marker.scale.y = marker.scale.z = 2;
      marker.pose.orientation.w = 1;
      marker.color.a = 1;
      marker.color.g = 1;
      marker.pose.position.x = local_goal[0];
      marker.pose.position.y = local_goal[1];
      marker.pose.position.z = 0;
      goal_viz.markers.push_back(marker);
    }
  }

  local_goal_viz_pub_.publish(goal_viz);
}

void LocalWrapper::printTimings() {
  ROS_INFO_STREAM("\033[34m" << "[SPOMP-Local]" << std::endl << 
      TimerManager::getGlobal() << "\033[0m");
}

} // namespace spomp
