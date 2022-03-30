#include "spomp/local_wrapper.h"
#include <iomanip>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>

namespace spomp {

LocalWrapper::LocalWrapper(ros::NodeHandle& nh) : 
  nh_(nh), 
  local_(createLocal(nh)), 
  remote_(50), 
  tf_buffer_(), 
  tf_listener_(tf_buffer_)
{
  auto& tm = TimerManager::getGlobal();
  viz_pano_t_ = tm.get("LW_viz_pano");
  viz_cloud_t_ = tm.get("LW_viz_cloud");

  obs_pano_viz_pub_ = nh_.advertise<sensor_msgs::Image>("obs_pano_viz", 1);
  obs_cloud_viz_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obs_cloud_viz", 1);
  reachability_viz_pub_ = nh_.advertise<sensor_msgs::LaserScan>("reachability_viz", 1);
}

Local LocalWrapper::createLocal(ros::NodeHandle& nh) {
  TerrainPano::Params tp_params{};
  PanoPlanner::Params pp_params{};

  nh.getParam("tbb", tp_params.tbb);
  nh.getParam("tbb", pp_params.tbb);

  nh.getParam("TP_max_hole_fill_size", tp_params.max_hole_fill_size);
  nh.getParam("TP_min_noise_size", tp_params.min_noise_size);
  nh.getParam("TP_v_fov_rad", tp_params.v_fov_rad);
  nh.getParam("TP_target_dist_xy", tp_params.target_dist_xy);
  nh.getParam("TP_noise_m", tp_params.noise_m);
  nh.getParam("TP_slope_thresh", tp_params.slope_thresh);
  nh.getParam("TP_inflation_m", tp_params.inflation_m);

  nh.getParam("PP_max_spacing_m", pp_params.max_spacing_m);

  using namespace std;
  ROS_INFO_STREAM("\033[32m" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(30) << "[ROS] tbb: " << tp_params.tbb << endl <<
    "[ROS] ===============================" << endl <<
    setw(30) << "[ROS] TP_max_hole_fill_size: " << tp_params.max_hole_fill_size << endl <<
    setw(30) << "[ROS] TP_min_noise_size: " << tp_params.min_noise_size << endl <<
    setw(30) << "[ROS] TP_v_fov_rad: " << tp_params.v_fov_rad << endl <<
    setw(30) << "[ROS] TP_target_dist_xy: " << tp_params.target_dist_xy << endl <<
    setw(30) << "[ROS] TP_noise_m: " << tp_params.noise_m << endl <<
    setw(30) << "[ROS] TP_slope_thresh: " << tp_params.slope_thresh << endl <<
    setw(30) << "[ROS] TP_inflation_m: " << tp_params.inflation_m << endl <<
    "[ROS] ===============================" << endl <<
    setw(30) << "[ROS] PP_max_spacing_m: " << pp_params.max_spacing_m << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");

  return Local(tp_params, pp_params);
}

void LocalWrapper::play() {
  use_tf_ = false;
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
    if (m.getTopic() == "/os_node/llol_odom/pano") {
      sensor_msgs::Image::ConstPtr pano = m.instantiate<sensor_msgs::Image>();
      if (pano != NULL) {
        if (!remote_.wait()) {
          // Want to quit
          break;
        }

        panoCallback(pano);
      }
    }
    ros::spinOnce();
  }
}

void LocalWrapper::initialize() {
  use_tf_ = true;
  pano_sub_ = nh_.subscribe<sensor_msgs::Image>("pano", 1, &LocalWrapper::panoCallback, this);

  ros::spin();
}

void LocalWrapper::panoCallback(const sensor_msgs::Image::ConstPtr& img_msg) {
  pano_frame_ = img_msg->header.frame_id;

  Eigen::MatrixXf pano_eig;
  cv_bridge::CvImageConstPtr pano_cv = cv_bridge::toCvShare(img_msg);
  cv::Mat depth_pano_cv(pano_cv->image.rows, pano_cv->image.cols, CV_16UC1);
  constexpr int from_to[] = {0, 0};
  // Extract first channel (depth)
  cv::mixChannels(&(pano_cv->image), 1, &depth_pano_cv, 1, from_to, 1);
  // Convert to Eigen and scale to meters
  cv::cv2eigen(depth_pano_cv, pano_eig);
  pano_eig /= 512;

  geometry_msgs::TransformStamped pano_pose;
  if (use_tf_) {
    try {
      pano_pose = tf_buffer_.lookupTransform("odom", pano_frame_, 
                                             img_msg->header.stamp);
    } catch (tf2::TransformException& ex) {
      ROS_WARN_STREAM("TF error: No transform found to pano");
      return;
    }
  }
  local_.updatePano(pano_eig, ROS2Eigen(pano_pose));

  visualizePano(img_msg->header.stamp);
  visualizeCloud(img_msg->header.stamp);
  visualizeReachability(img_msg->header.stamp);
  printTimings();
}

void LocalWrapper::visualizePano(const ros::Time& stamp) {
  viz_pano_t_->start();

  const Eigen::MatrixXi& pano = local_.getPano().getTraversability().matrix();
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
  cloud_packed.row(3) = Eigen::Map<const Eigen::VectorXi>(
      trav.data(), trav.size()).cast<float>();

  obs_cloud_viz_pub_.publish(cloud_msg);

  viz_cloud_t_->end();
}

void LocalWrapper::visualizeReachability(const ros::Time& stamp) {
  const auto& reachability = local_.getPlanner().getReachability();
  const auto& azs = local_.getPano().getAzs();

  sensor_msgs::LaserScan scan_msg;
  scan_msg.header.frame_id = pano_frame_;
  scan_msg.header.stamp = stamp;

  // Order is flipped, since all negatives
  scan_msg.angle_min = azs[azs.size()-1] + deg2rad(360);
  scan_msg.angle_max = azs[0] + deg2rad(360);
  scan_msg.angle_increment = azs[0] - azs[1];
  scan_msg.range_max = 100; // Something large

  scan_msg.ranges.resize(azs.size());
  Eigen::Map<Eigen::VectorXf> scan_ranges(reinterpret_cast<float*>(scan_msg.ranges.data()),
      azs.size());
  // Order flips here again
  scan_ranges = reachability.reverse();

  reachability_viz_pub_.publish(scan_msg);
}

void LocalWrapper::printTimings() {
  ROS_INFO_STREAM("\033[34m" << TimerManager::getGlobal() << "\033[0m");
}

Eigen::Isometry3f LocalWrapper::ROS2Eigen(const geometry_msgs::TransformStamped& trans_msg) {
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translate(Eigen::Vector3f(
        trans_msg.transform.translation.x,
        trans_msg.transform.translation.y,
        trans_msg.transform.translation.z
        ));
  pose.rotate(Eigen::Quaternionf(
        trans_msg.transform.rotation.w,
        trans_msg.transform.rotation.x,
        trans_msg.transform.rotation.y,
        trans_msg.transform.rotation.z
        ));
  return pose;
}

geometry_msgs::TransformStamped LocalWrapper::Eigen2ROS(const Eigen::Isometry3f& pose) {
  geometry_msgs::TransformStamped pose_msg;
  pose_msg.transform.translation.x = pose.translation()[0];
  pose_msg.transform.translation.y = pose.translation()[1];
  pose_msg.transform.translation.z = pose.translation()[2];
  Eigen::Quaternionf quat(pose.rotation());
  pose_msg.transform.rotation.x = quat.x();
  pose_msg.transform.rotation.y = quat.y();
  pose_msg.transform.rotation.z = quat.z();
  pose_msg.transform.rotation.w = quat.w();

  return pose_msg;
}

} // namespace spomp
