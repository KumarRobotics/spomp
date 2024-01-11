#include "spomp/mapper_wrapper.h"
#include "spomp/rosutil.h"
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>

namespace spomp {

std::string MapperWrapper::odom_frame_{"odom"};
std::string MapperWrapper::map_frame_{"map"};
int MapperWrapper::viz_thread_period_ms_{1000};

/**
 * @class MapperWrapper
 * @brief Wrapper class for a ROS node that implements mapping functionality.
 *
 * This class provides a wrapper for a ROS node that performs mapping using different library functions.
 * It initializes the necessary components, subscribes to relevant topics, and publishes mapping results.
 */
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
  publishOdomCorrection(Eigen::Isometry3d::Identity(), ros::Time::now());
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

/**
 * @brief Initializes the MapperWrapper class.
 *
 * This function sets up the subscribers and timers used by the MapperWrapper class.
 * - Subscribes to the "pano/img" topic with camera_info.
 * - Subscribes to the "global_est" topic.
 * - Subscribes to the "pose" topic.
 * - Subscribes to the "pano/sem" topic.
 * - Creates a timer for visualization, with a period specified by viz_thread_period_ms_.
 * - Calls the ros::spin() function to start the ROS event loop.
 */
    void MapperWrapper::initialize() {
  // Subscribers
  pano_sub_ = it_.subscribeCamera("pano/img", 1, &MapperWrapper::panoCallback, this);
  est_sub_ = nh_.subscribe("global_est", 1, &MapperWrapper::globalEstCallback, this);
  odom_sub_ = nh_.subscribe("pose", 1, &MapperWrapper::odomCallback, this);
  sem_pano_sub_ = nh_.subscribe("pano/sem", 1, &MapperWrapper::semPanoCallback, this);

  // Timers
  viz_timer_ = nh_.createTimer(ros::Duration(viz_thread_period_ms_/1000.), 
      &MapperWrapper::visualize, this);

  ros::spin();
}

/**
* @brief Function that is called when a new panoramic image is received
*
* This function takes in a sensor_msgs::Image and a sensor_msgs::CameraInfo and processes them to create a keyframe
* for the mapper. The function extracts the camera pose from the CameraInfo message, converts the input image to OpenCV format,
* splits the image into separate channels, rescales the depth channel using the depth scale from the CameraInfo message,
* and finally adds the keyframe to the Mapper.
*
* @param img_msg  Pointer to the sensor_msgs::Image message that contains the panoramic image
* @param info_msg  Pointer to the sensor_msgs::CameraInfo message that contains the camera information
*
* @return void
*/
    void MapperWrapper::panoCallback(const sensor_msgs::Image::ConstPtr& img_msg,
                                     const sensor_msgs::CameraInfo::ConstPtr& info_msg)
{
  Eigen::Isometry3d pano_pose;
  pano_pose.affine() = 
    Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(&info_msg->P[0]);

  auto pano = cv_bridge::toCvShare(img_msg);
  std::vector<cv::Mat> channels;
  channels.resize(pano->image.channels());
  cv::split(pano->image, channels);

  float depth_scale = info_msg->R[0];
  cv::Mat rescaled_depth;
  channels[0].convertTo(rescaled_depth, CV_32F, 1./depth_scale);

  mapper_.addKeyframe({info_msg->header.stamp.toNSec(), 
                       pano_pose,
                       rescaled_depth,
                       channels[1], /* intensity */
                       cv::Mat() /* semantics (empty for now) */});
}

/**
 * @brief globalEstCallback function for receiving global estimation callback
 *
 * This function is called whenever a global estimation message is received.
 * It processes the received message and updates the mapper object with the prior information.
 *
 * @param est_msg The received global estimation message
 */
    void MapperWrapper::globalEstCallback(
    const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& est_msg)
{
  if (!initialized_odom_corr_) {
    // Special case for first estimate
    publishOdomCorrection(ROS2Eigen<double>(est_msg->pose.pose), est_msg->header.stamp);

    initialized_odom_corr_ = true;
  }

  // Loop starting with most recent
  // We assume that the global est will arrive after odom
  // This is fairly reasonable, since global localizer uses odom as input
  for (auto odom_it = odom_buf_.rbegin(); odom_it != odom_buf_.rend(); ++odom_it) {
    if ((*odom_it)->header.stamp == est_msg->header.stamp) {
      Mapper::StampedPrior prior{};
      prior.stamp = est_msg->header.stamp.toNSec();
      prior.prior.local_pose = ROS2Eigen<double>(**odom_it);
      prior.prior.pose = pose32pose2(ROS2Eigen<double>(est_msg->pose.pose));

      // organized as pos, rot, flipped from gtsam
      const auto cov = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(
          &est_msg->pose.covariance[0]); 
      // For now, just take the diagonal
      // In the future, could use the full covariance
      prior.prior.sigma_diag[0] = std::sqrt(cov.diagonal()[5]);
      prior.prior.sigma_diag.tail<2>() = cov.diagonal().head<2>().array().sqrt();

      mapper_.addPrior(prior);
      publishOdomCorrection(mapper_.getOdomCorrection(), est_msg->header.stamp);

      odom_buf_.erase(odom_buf_.begin(), odom_it.base());
      break;
    }
  }
}

/**
 * @brief Callback function for the odometry topic
 *
 * This function is a callback for the odometry topic and stores the received odometry message in the odom_buf_ buffer.
 *
 * @param[in] odom_msg Pointer to the received odometry message
 */
    void MapperWrapper::odomCallback(
    const geometry_msgs::PoseStamped::ConstPtr &odom_msg)
{
  odom_buf_.push_back(odom_msg);
}

/**
 * @brief This function is a callback function for the semPano topic.
 *        It receives a sensor_msgs::Image::ConstPtr as a parameter, and performs the following actions:
 *        - Converts the received image message to CV image format using cv_bridge::toCvCopy()
 *        - Calls the addSemantics() function of the mapper_ object, passing the timestamp and the converted image
 * @param sem_img_msg A constant pointer to the received sensor_msgs::Image message
 * @return void
 */
    void MapperWrapper::semPanoCallback(const sensor_msgs::Image::ConstPtr& sem_img_msg) {
  auto sem_pano = cv_bridge::toCvCopy(
      sem_img_msg, sensor_msgs::image_encodings::MONO8);
  mapper_.addSemantics({sem_img_msg->header.stamp.toNSec(), sem_pano->image});
}

/**
 * @brief Visualize method for the MapperWrapper class.
 *
 * This method is triggered by a timer event. It starts the visualization and publishes the grid map message.
 * It also visualizes the pose graph and prints the global timer information.
 *
 * @param timer The timer event information.
 */
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

/**
 * @brief Visualize the pose graph.
 *
 * This function creates markers for visualizing the trajectory and key poses in the pose graph.
 *
 * @param stamp The timestamp to be used for the marker headers.
 */
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

/**
 * @brief Publishes odometry correction
 *
 * This function publishes the odometry correction as a transform
 * message using the ROS tf library. It takes the corrected odometry
 * transform (corr) and the timestamp (stamp) as input parameters.
 *
 * @param corr The corrected odometry transform
 * @param stamp The timestamp of the correction
 */
    void MapperWrapper::publishOdomCorrection(const Eigen::Isometry3d& corr,
                                              const ros::Time& stamp)
{
  geometry_msgs::TransformStamped corr_msg = Eigen2ROS(corr);
  corr_msg.header.stamp = stamp;
  corr_msg.header.frame_id = map_frame_;
  corr_msg.child_frame_id = odom_frame_;
  tf_static_broadcaster_.sendTransform(corr_msg);
}

} // namespace spomp
