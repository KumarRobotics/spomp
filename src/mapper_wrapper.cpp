#include "spomp/mapper_wrapper.h"
#include "spomp/rosutil.h"
#include <visualization_msgs/MarkerArray.h>

namespace spomp {

MapperWrapper::MapperWrapper(ros::NodeHandle& nh) : 
  nh_(nh),
  it_(nh),
  mapper_(createMapper(nh)) 
{
  auto& tm = TimerManager::getGlobal(true);
  viz_t_ = tm.get("MW_viz");

  // Publishers
  graph_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("graph_viz", 1);
}

Mapper MapperWrapper::createMapper(ros::NodeHandle& nh) {
  Mapper::Params m_params;
  PoseGraph::Params pg_params;

  return Mapper(m_params, pg_params);
}

void MapperWrapper::initialize() {
  // Subscribers
  pano_sub_ = it_.subscribeCamera("pano/img", 1, &MapperWrapper::panoCallback, this);
  est_sub_ = std::make_unique<message_filters::Subscriber<
    geometry_msgs::PoseWithCovarianceStamped>>(nh_, "global_est", 1);
  odom_sub_ = std::make_unique<message_filters::Subscriber<
    geometry_msgs::PoseStamped>>(nh_, "odom", 1);
  global_est_odom_sync_ = std::make_unique<message_filters::TimeSynchronizer<
    geometry_msgs::PoseWithCovarianceStamped, 
    geometry_msgs::PoseStamped>>(*est_sub_, *odom_sub_, 10);
  global_est_odom_sync_->registerCallback(&MapperWrapper::globalEstCallback, this);

  // Timers
  viz_timer_ = nh_.createTimer(ros::Duration(1), &MapperWrapper::visualize, this);

  ros::spin();
}

void MapperWrapper::panoCallback(const sensor_msgs::Image::ConstPtr& img_msg,
    const sensor_msgs::CameraInfo::ConstPtr& info_msg) 
{
  Eigen::Isometry3d pano_pose;
  pano_pose.affine() = 
    Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(&info_msg->P[0]);
  mapper_.addKeyframe({static_cast<long>(info_msg->header.stamp.toNSec()), 
                       pano_pose});
}

void MapperWrapper::globalEstCallback(
    const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& est_msg,
    const geometry_msgs::PoseStamped::ConstPtr &odom_msg)
{
  Mapper::StampedPrior prior{};
  prior.stamp = est_msg->header.stamp.toNSec();
  prior.prior.local_pose = ROS2Eigen<double>(*odom_msg);
  prior.prior.pose = pose32pose2(ROS2Eigen<double>(est_msg->pose.pose));

  // organized as pos, rot, flipped from gtsam
  const auto cov = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(
      &est_msg->pose.covariance[0]); 
  // For now, just take the diagonal
  // In the future, could use the full covariance
  prior.prior.sigma_diag[0] = std::sqrt(cov.diagonal()[5]);
  prior.prior.sigma_diag.tail<2>() = cov.diagonal().head<2>().array().sqrt();

  mapper_.addPrior(prior);
}

void MapperWrapper::visualize(const ros::TimerEvent& timer) {
  viz_t_->start();
  ros::Time stamp;
  stamp.fromNSec(mapper_.stamp());
  vizPoseGraph(stamp);
  publishOdomCorrection(stamp);
  viz_t_->end();

  ROS_INFO_STREAM("\033[34m" << TimerManager::getGlobal(true) << "\033[0m");
}

void MapperWrapper::vizPoseGraph(const ros::Time& stamp) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker traj_marker, key_marker;

  traj_marker.header.frame_id = "map";
  traj_marker.header.stamp = stamp;
  traj_marker.ns = "trajectory";
  traj_marker.id = 0;
  traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
  traj_marker.action = visualization_msgs::Marker::ADD;
  traj_marker.pose.position.x = 0;
  traj_marker.pose.position.y = 0;
  traj_marker.pose.position.z = 0;
  traj_marker.pose.orientation.x = 0;
  traj_marker.pose.orientation.y = 0;
  traj_marker.pose.orientation.z = 0;
  traj_marker.pose.orientation.w = 1;
  traj_marker.scale.x = 0.2; // Line width
  traj_marker.color.a = 1;
  traj_marker.color.r = 1;
  traj_marker.color.g = 0;
  traj_marker.color.b = 0;

  key_marker.header = traj_marker.header;
  key_marker.ns = "keys";
  key_marker.id = 0;
  key_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  key_marker.action = visualization_msgs::Marker::ADD;
  key_marker.pose = traj_marker.pose;
  key_marker.scale.x = 1;
  key_marker.scale.y = 1;
  key_marker.scale.z = 1;
  key_marker.color.a = 1;
  key_marker.color.r = 0;
  key_marker.color.g = 1;
  key_marker.color.b = 0;

  auto graph = mapper_.getGraph();
  for (const auto& pose : graph) {
    auto pt_msg = Eigen2ROS<double>(pose.translation());
    traj_marker.points.push_back(pt_msg);
    key_marker.points.push_back(pt_msg);
  }

  marker_array.markers.push_back(traj_marker);
  marker_array.markers.push_back(key_marker);

  graph_viz_pub_.publish(marker_array);
}

void MapperWrapper::publishOdomCorrection(const ros::Time& stamp) {
  Eigen::Isometry3d corr = mapper_.getOdomCorrection();
  geometry_msgs::TransformStamped corr_msg = Eigen2ROS(corr);
  corr_msg.header.stamp = stamp;
  corr_msg.header.frame_id = "map";
  corr_msg.child_frame_id = "odom";
  static_tf_broadcaster_.sendTransform(corr_msg);
}

} // namespace spomp
