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
  viz_t_->end();

  ROS_INFO_STREAM("\033[34m" << TimerManager::getGlobal(true) << "\033[0m");
}

} // namespace spomp
