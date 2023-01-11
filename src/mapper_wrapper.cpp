#include "spomp/mapper_wrapper.h"
#include "spomp/rosutil.h"
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>

namespace spomp {

std::string MapperWrapper::odom_frame_{"odom"};
std::string MapperWrapper::map_frame_{"map"};
int MapperWrapper::viz_thread_period_ms_{1000};

MapperWrapper::MapperWrapper(ros::NodeHandle& nh) : 
  nh_(nh),
  it_(nh),
  mapper_(createMapper(nh)) 
{
  auto& tm = TimerManager::getGlobal(true);
  viz_t_ = tm.get("MW_viz");

  // Publishers
  graph_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("graph_viz", 1);
  map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("map", 1);

  // Publish initial ident transform
  publishOdomCorrection(ros::Time::now());
}

Mapper MapperWrapper::createMapper(ros::NodeHandle& nh) {
  Mapper::Params m_params{};
  PoseGraph::Params pg_params{};
  MetricMap::Params mm_params{};

  nh.getParam("world_config_path", mm_params.world_config_path);

  nh.getParam("odom_frame", odom_frame_);
  nh.getParam("map_frame", map_frame_);
  nh.getParam("viz_thread_period_ms", viz_thread_period_ms_);

  nh.getParam("M_pgo_thread_period_ms", m_params.pgo_thread_period_ms);
  nh.getParam("M_map_thread_period_ms", m_params.map_thread_period_ms);
  nh.getParam("M_correct_odom_per_frame", m_params.correct_odom_per_frame);
  nh.getParam("M_dist_between_keyframes_m", m_params.dist_between_keyframes_m);
  nh.getParam("M_pano_v_fov_rad", m_params.pano_v_fov_rad);
  nh.getParam("M_require_sem", m_params.require_sem);

  nh.getParam("PG_num_frames_opt", pg_params.num_frames_opt);
  nh.getParam("PG_allow_interpolation", pg_params.allow_interpolation);
  float loc, rot;
  bool have_unc = false;
  if (nh.getParam("PG_between_uncertainty_loc", loc) &&
      nh.getParam("PG_between_uncertainty_rot", rot)) {
    pg_params.setBetweenUncertainty(loc, rot);
  }
  have_unc = false;
  if (nh.getParam("PG_prior_uncertainty_loc", loc) &&
      nh.getParam("PG_prior_uncertainty_rot", rot)) {
    pg_params.setPriorUncertainty(loc, rot);
  }

  nh.getParam("MM_resolution", mm_params.resolution);
  nh.getParam("MM_buffer_size_m", mm_params.buffer_size_m);
  nh.getParam("MM_req_point_density", mm_params.req_point_density);
  nh.getParam("MM_dist_for_rebuild_m", mm_params.dist_for_rebuild_m);
  nh.getParam("MM_ang_for_rebuild_rad", mm_params.ang_for_rebuild_rad);

  constexpr int width = 30;
  using namespace std;
  ROS_INFO_STREAM("\033[32m" << "[SPOMP-Mapper]" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(width) << "[ROS] world_config_path: " << mm_params.world_config_path << endl <<
    setw(width) << "[ROS] odom_frame: " << odom_frame_ << endl <<
    setw(width) << "[ROS] map_frame: " << map_frame_ << endl <<
    setw(width) << "[ROS] viz_thread_period_ms: " << viz_thread_period_ms_ << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] M_pgo_thread_period_ms: " << m_params.pgo_thread_period_ms << endl <<
    setw(width) << "[ROS] M_map_thread_period_ms: " << m_params.map_thread_period_ms << endl <<
    setw(width) << "[ROS] M_correct_odom_per_frame: " << m_params.correct_odom_per_frame << endl <<
    setw(width) << "[ROS] M_dist_between_keyframes_m: " << m_params.dist_between_keyframes_m << endl <<
    setw(width) << "[ROS] M_pano_v_fov_rad: " << m_params.pano_v_fov_rad << endl <<
    setw(width) << "[ROS] M_require_sem: " << m_params.require_sem << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] PG_num_frames_opt: " << pg_params.num_frames_opt << endl <<
    setw(width) << "[ROS] PG_allow_interpolation: " << pg_params.allow_interpolation << endl <<
    setw(width) << "[ROS] PG_between_uncertainty: " << pg_params.between_uncertainty.transpose() << endl <<
    setw(width) << "[ROS] PG_prior_uncertainty: " << pg_params.prior_uncertainty.transpose() << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] MM_resolution: " << mm_params.resolution << endl <<
    setw(width) << "[ROS] MM_buffer_size_m: " << mm_params.buffer_size_m << endl <<
    setw(width) << "[ROS] MM_req_point_density: " << mm_params.req_point_density << endl <<
    setw(width) << "[ROS] MM_dist_for_rebuild_m: " << mm_params.dist_for_rebuild_m << endl <<
    setw(width) << "[ROS] MM_ang_for_rebuild_rad: " << mm_params.ang_for_rebuild_rad << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");

  return Mapper(m_params, pg_params, mm_params);
}

void MapperWrapper::initialize() {
  // Subscribers
  pano_sub_ = it_.subscribeCamera("pano/img", 1, &MapperWrapper::panoCallback, this);
  est_sub_ = std::make_unique<message_filters::Subscriber<
    geometry_msgs::PoseWithCovarianceStamped>>(nh_, "global_est", 5);
  odom_sub_ = std::make_unique<message_filters::Subscriber<
    geometry_msgs::PoseStamped>>(nh_, "pose", 100);
  global_est_odom_sync_ = std::make_unique<message_filters::TimeSynchronizer<
    geometry_msgs::PoseWithCovarianceStamped, 
    geometry_msgs::PoseStamped>>(*est_sub_, *odom_sub_, 100);
  global_est_odom_sync_->registerCallback(&MapperWrapper::globalEstCallback, this);
  sem_pano_sub_ = nh_.subscribe("pano/sem", 1, &MapperWrapper::semPanoCallback, this);

  // Timers
  viz_timer_ = nh_.createTimer(ros::Duration(viz_thread_period_ms_/1000.), 
      &MapperWrapper::visualize, this);

  ros::spin();
}

void MapperWrapper::panoCallback(const sensor_msgs::Image::ConstPtr& img_msg,
    const sensor_msgs::CameraInfo::ConstPtr& info_msg) 
{
  Eigen::Isometry3d pano_pose;
  pano_pose.affine() = 
    Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(&info_msg->P[0]);

  auto pano = cv_bridge::toCvShare(
      img_msg, sensor_msgs::image_encodings::TYPE_16UC3);
  std::vector<cv::Mat> channels;
  channels.resize(pano->image.channels());
  cv::split(pano->image, channels);

  float depth_scale = info_msg->R[0];
  cv::Mat rescaled_depth;
  channels[0].convertTo(rescaled_depth, CV_32F, 1./depth_scale);

  mapper_.addKeyframe({info_msg->header.stamp.toNSec(), 
                       pano_pose,
                       rescaled_depth,
                       channels[1],
                       cv::Mat()});
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

  publishOdomCorrection(est_msg->header.stamp);
  mapper_.addPrior(prior);
}

void MapperWrapper::semPanoCallback(const sensor_msgs::Image::ConstPtr& sem_img_msg) {
  auto sem_pano = cv_bridge::toCvCopy(
      sem_img_msg, sensor_msgs::image_encodings::MONO8);
  mapper_.addSemantics({sem_img_msg->header.stamp.toNSec(), sem_pano->image});
}

void MapperWrapper::visualize(const ros::TimerEvent& timer) {
  viz_t_->start();
  ros::Time stamp;
  stamp.fromNSec(mapper_.stamp());
  vizPoseGraph(stamp);
  map_pub_.publish(mapper_.getGridMapMsg());
  viz_t_->end();

  ROS_INFO_STREAM("\033[34m" << "[SPOMP-Mapper]" << std::endl << 
      TimerManager::getGlobal(true) << "\033[0m");
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
  corr_msg.header.frame_id = map_frame_;
  corr_msg.child_frame_id = odom_frame_;
  tf_static_broadcaster_.sendTransform(corr_msg);
}

} // namespace spomp
