#include "spomp/mapper.h"
#include "spomp/utils.h"

namespace spomp {

/**
 * @class Mapper
 * @brief The Mapper class represents a mapper object that creates a map based on received sensor data.
 */
    Mapper::Mapper(const Params& m_p, const PoseGraph::Params& pg_p,
                   const MetricMap::Params& mm_p)
{
  pose_graph_thread_ = std::thread(PoseGraphThread(*this, {pg_p}));
  map_thread_ = std::thread(MapThread(*this, {mm_p}));
}

/**
* @brief Destructor for the Mapper class.
*
* This destructor is responsible for cleaning up the Mapper object. It sets the exit_threads_flag_
* to true, which signals the pose_graph_thread_ and map_thread_ to exit. It then joins the threads,
* allowing them to complete any remaining work before the Mapper object is destroyed.
*/
    Mapper::~Mapper() {
  exit_threads_flag_ = true;
  pose_graph_thread_.join();
  map_thread_.join();
}

/**
 * @brief Adds a new keyframe to the Mapper.
 *
 * This function checks if the new keyframe is far enough from the last keyframe in terms of translation.
 * If the distance between the poses of the new keyframe and the last keyframe is greater than the specified distance,
 * the intrinsic parameters of the Keyframe class are set, the last keyframe pose is updated to the pose of the new keyframe,
 * and the new keyframe is added to the list of frames in the keyframe_input_ struct.
 *
 * @param k The keyframe to be added.
 */
    void Mapper::addKeyframe(const Keyframe& k) {
  if ((k.getPose().inverse() * last_keyframe_pose_).translation().norm() > 
      params_.dist_between_keyframes_m) 
  {
    Keyframe::setIntrinsics(params_.pano_v_fov_rad, k.getSize());
    last_keyframe_pose_ = k.getPose();
    std::scoped_lock lock(keyframe_input_.mtx);
    keyframe_input_.frames.emplace_back(k);
  }
}

/**
 * @brief Adds a StampedPrior to the mapper's list of priors.
 *
 * @param p The StampedPrior to be added.
 */
    void Mapper::addPrior(const StampedPrior& p) {
  last_prior_ = p;
  std::scoped_lock lock(prior_input_.mtx);
  prior_input_.priors.emplace_back(p);
}

/**
 * @brief Adds a StampedSemantics object to the list of sem_panos in the SemanticsInput structure.
 *
 * @param s The StampedSemantics object to be added.
 */
    void Mapper::addSemantics(const StampedSemantics& s) {
  std::scoped_lock lock(semantics_input_.mtx);
  semantics_input_.sem_panos.emplace_back(s);
}

std::vector<Eigen::Isometry3d> Mapper::getGraph() { 
  std::vector<Eigen::Isometry3d> poses;
  std::shared_lock key_lock(keyframes_.mtx);
  poses.reserve(keyframes_.frames.size());

  for (const auto& frame : keyframes_.frames) {
    poses.emplace_back(frame.second.getPose());
  }

  return poses;
}

/**
 * @brief Function to retrieve the odometry correction
 *
 * This function returns the odometry correction as an Eigen::Isometry3d object.
 * The correction is determined based on the value of the 'correct_odom_per_frame'
 * flag in the params_ object.
 *
 * If 'correct_odom_per_frame' is not set, the function retrieves the stored
 * odometry correction from the keyframes_.odom_corr member variable and returns it.
 *
 * If 'correct_odom_per_frame' is set, the function calculates the odometry correction
 * based on the last prior pose and the inverse of the last prior local pose. The
 * result is then returned.
 *
 * @return Eigen::Isometry3d - The odometry correction
 */
    Eigen::Isometry3d Mapper::getOdomCorrection() {
  Eigen::Isometry3d corr;
  if (!params_.correct_odom_per_frame) {
    std::shared_lock key_lock(keyframes_.mtx);
    corr = keyframes_.odom_corr;
  } else {
    corr = pose22pose3(last_prior_.prior.pose) * 
           last_prior_.prior.local_pose.inverse();
  }
  return corr;
}

/**
 * @brief Get the most recent timestamp of the keyframes.
 *
 * This function retrieves the timestamp of the latest keyframe in the keyframes container.
 * If there are no keyframes, it will return 0.
 *
 * @return The timestamp of the most recent keyframe, or 0 if there are no keyframes.
 */
    long Mapper::stamp() {
  std::shared_lock key_lock(keyframes_.mtx);
  if (keyframes_.frames.size() < 1) return 0;
  return keyframes_.frames.rbegin()->first;
}

/**
 * @brief This function is the main loop of a thread that performs pose graph operations.
 *
 * The function runs a loop until the `exit_threads_flag_` of the `mapper_` object is set to true.
 * Inside the loop, the function calls several helper functions to perform various operations on the pose graph.
 *
 * @return Always returns true.
 */
bool Mapper::PoseGraphThread::operator()() {
  auto& tm = TimerManager::getGlobal(true);
  parse_buffer_t_ = tm.get("PG_parse_buffer");
  parse_semantics_buffer_t_ = tm.get("PG_parse_semantics_buffer");
  update_keyframes_t_ = tm.get("PG_update_keyframes");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    parseBuffer(); 
    parseSemanticsBuffer();
    pg_.update();
    updateKeyframes();

    // Sleep until next loop
    next += milliseconds(mapper_.params_.pgo_thread_period_ms);
    if (next < steady_clock::now()) {
      next = steady_clock::now();
    } else {
      std::this_thread::sleep_until(next);
    }
  }

  return true;
}

/**
 * @brief Parses the keyframe and prior buffers and updates the PoseGraph.
 *
 * This function parses the keyframe buffer and the prior buffer, and updates the PoseGraph accordingly.
 * After parsing the keyframe buffer, it adds the keyframes to the PoseGraph, updates their poses based on relative motion,
 * and adds them to the keyframes container in the Mapper object.
 * After parsing the prior buffer, it adds the priors to the PoseGraph.
 */
    void Mapper::PoseGraphThread::parseBuffer() {
  parse_buffer_t_->start();
  {
    // Service keyframe buffer
    std::scoped_lock lock(mapper_.keyframe_input_.mtx);

    Eigen::Isometry3d pg_pose;
    for (auto& frame : mapper_.keyframe_input_.frames) {
      auto ind = pg_.addNode(frame.getStamp(), frame.getPose());
      // PoseGraph updates pose on addition based on relative motion
      pg_pose = pg_.getPoseAtIndex(ind);
      frame.setPose(pg_pose);

      std::unique_lock key_lock(mapper_.keyframes_.mtx);
      mapper_.keyframes_.frames.insert({frame.getStamp(), std::move(frame)});
    }

    mapper_.keyframe_input_.frames.clear();
  }
  {
    // Service prior buffer
    std::scoped_lock lock(mapper_.prior_input_.mtx);

    for (auto& prior : mapper_.prior_input_.priors) {
      pg_.addPrior(prior.stamp, std::move(prior.prior));
    }

    mapper_.prior_input_.priors.clear();
  }
  parse_buffer_t_->end();
}

/**
 * @brief Parses the semantic buffer and matches semantic panos to corresponding keyframes.
 *
 * This function iterates through the semantic panos in the input buffer and tries to match them with
 * the corresponding keyframes in the mapper's keyframe list. If a match is found, the semantic information
 * is set on the keyframe. If no match is found, the semantic pano is removed from the buffer.
 *
 * @note It is assumed that semantic panos will always arrive later than the corresponding keyframes.
 *
 * @note Locks are used to ensure thread safety while accessing the semantic buffer and keyframes.
 */
    void Mapper::PoseGraphThread::parseSemanticsBuffer() {
  parse_semantics_buffer_t_->start();

  std::scoped_lock sem_lock(mapper_.semantics_input_.mtx);

  // We don't need a lock on keyframes for reading, since this thread is the only
  // thread that could add keyframes, and stamps are constant
  for (auto sem_it = mapper_.semantics_input_.sem_panos.begin();
       sem_it != mapper_.semantics_input_.sem_panos.end();) {
    auto key_it = mapper_.keyframes_.frames.find(sem_it->stamp);
    if (key_it != mapper_.keyframes_.frames.end()) {
      // We have a matching keyframe
      {
        std::unique_lock key_lock(mapper_.keyframes_.mtx);
        key_it->second.setSem(sem_it->pano);
      }
      // Advance by removing from buffer
      sem_it = mapper_.semantics_input_.sem_panos.erase(sem_it);
    } else if (mapper_.keyframes_.frames.size() > 0) {
      if (sem_it->stamp < mapper_.keyframes_.frames.rbegin()->second.getStamp()) {
        // The sem pano is older than newest images in keyframe list
        // We assume that semantic panos will always arrive later
        sem_it = mapper_.semantics_input_.sem_panos.erase(sem_it);
      } else {
        ++sem_it;
      }
    } else {
      // No keyframes yet, so no point trying to match
      break;
    }
  }

  parse_semantics_buffer_t_->end();
}

/**
 * \brief Update keyframes in the pose graph.
 *
 * This function updates the poses of the keyframes in the pose graph based on the latest information
 * from the pose graph. It retrieves the poses at each keyframe timestamp from the pose graph and sets
 * the corresponding keyframe pose in the frames container. It also updates the odometry correction in
 * the keyframes container.
 *
 * \note This function assumes that the update_keyframes_t_ timer has been started before calling this function.
 */
    void Mapper::PoseGraphThread::updateKeyframes() {
  update_keyframes_t_->start();
  std::unique_lock key_lock(mapper_.keyframes_.mtx);

  for (auto& key : mapper_.keyframes_.frames) {
    auto new_pose = pg_.getPoseAtTime(key.first);
    if (new_pose) {
      key.second.setPose(*new_pose);
      key.second.setOptimized();
    }
  }
  mapper_.keyframes_.odom_corr = pg_.getOdomCorrection();

  update_keyframes_t_->end();
}

/**
 * @brief The MapThread class is responsible for mapping the keyframes and updating the map.
 */
bool Mapper::MapThread::operator()() {
  auto& tm = TimerManager::getGlobal(true);
  get_keyframes_to_compute_t_ = tm.get("M_get_keyframes_to_compute");
  resize_map_t_ = tm.get("M_resize_map");
  update_map_t_ = tm.get("M_update_map");
  export_map_t_ = tm.get("M_export_map");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    auto keyframes_to_compute = getKeyframesToCompute();
    if (keyframes_to_compute.size() > 0) {
      resizeMap(keyframes_to_compute);
      updateMap(keyframes_to_compute);
      exportMap();
    }

    // Sleep until next loop
    next += milliseconds(mapper_.params_.map_thread_period_ms);
    if (next < steady_clock::now()) {
      next = steady_clock::now();
    } else {
      std::this_thread::sleep_until(next);
    }
  }

  return true;
}

/**
 * @brief Retrieves a vector of keyframes to be computed.
 *
 * This function retrieves keyframes from the `keyframes_` container in the `Mapper` class
 * that need to be computed. Keyframes are selected based on whether they require a map update,
 * are optimized, and have semantic data (if required). The function also checks if the map needs
 * to be rebuilt, in which case all keyframes are considered. The map pose of each selected keyframe
 * is updated. A copy of each selected keyframe is added to the `keyframes_to_compute` vector.
 *
 * If the map needs to be rebuilt, the `keyframes_to_compute` vector is cleared and the `map_`
 * is cleared. All keyframes are then added to the `keyframes_to_compute` vector after updating their
 * map poses.
 *
 * @return The vector of keyframes to be computed.
 */
    std::vector<Keyframe> Mapper::MapThread::getKeyframesToCompute() {
  get_keyframes_to_compute_t_->start();
  // Shared lock because we are updating map pose in keyframe, but we only ever do
  // that in this thread, so still safe to be "read-only"
  std::shared_lock lock(mapper_.keyframes_.mtx);

  std::vector<Keyframe> keyframes_to_compute;
  bool rebuild_map = false;
  for (auto& frame : mapper_.keyframes_.frames) {
    if (map_.needsMapUpdate(frame.second) && frame.second.isOptimized() &&
        (frame.second.haveSem() || !mapper_.params_.require_sem)) {
      if (frame.second.inMap()) {
        rebuild_map = true;
        break;
      }
      frame.second.updateMapPose();

      // This is a copy, but not too bad since the big data stuff is
      // in cv::Mat, which is essentially a shared_ptr
      keyframes_to_compute.push_back(frame.second);
    }
  }

  if (rebuild_map) {
    keyframes_to_compute.clear();
    map_.clear();
    
    // Add all frames
    for (auto& frame : mapper_.keyframes_.frames) {
      if (frame.second.isOptimized()) {
        frame.second.updateMapPose();
        keyframes_to_compute.push_back(frame.second);
      }
    }
  }

  get_keyframes_to_compute_t_->end();
  return keyframes_to_compute;
}

/**
 * @brief Resizes the map based on the given frames.
 *
 * This method calculates the minimum and maximum coordinates of the frames and resizes the map to fit those coordinates.
 *
 * @param frames The vector of Keyframes.
 */
    void Mapper::MapThread::resizeMap(const std::vector<Keyframe>& frames) {
  resize_map_t_->start();

  Eigen::Vector2d min = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d max = Eigen::Vector2d::Constant(std::numeric_limits<double>::lowest());
  for (auto& frame : frames) {
    Eigen::Vector3d loc = frame.getPose().translation();
    min = min.cwiseMin(loc.head<2>());
    max = max.cwiseMax(loc.head<2>());
  }
  if (frames.size() > 0) {
    map_.resizeToBounds(min, max);
  }

  resize_map_t_->end();
}

/**
 * @brief Update the map with a vector of Keyframes.
 *
 * This function updates the map with the point clouds from
 * each keyframe in the given vector. It iterates over each
 * keyframe, retrieves its point cloud and timestamp, and
 * adds the cloud to the map using the addCloud() function of
 * the MetricMap class.
 *
 * @param frames A vector of Keyframes that contain the point
 *               clouds to be added to the map.
 */
    void Mapper::MapThread::updateMap(const std::vector<Keyframe>& frames) {
  update_map_t_->start();
  for (const auto& frame : frames) {
    map_.addCloud(frame.getPointCloud(), frame.getStamp());
  }
  update_map_t_->end();
}

/**
 * @brief Export the map to ROS message format
 *
 * This function exports the map to ROS message format using the exportROSMsg function of the MetricMap class.
 * It also locks the map_messages mutex to ensure thread safety when accessing and updating the map_messages grid_map.
 *
 * @note This function should be called from a separate thread.
 */
    void Mapper::MapThread::exportMap() {
  export_map_t_->start();
  {
    std::scoped_lock<std::mutex> lock(mapper_.map_messages_.mtx);
    mapper_.map_messages_.grid_map = map_.exportROSMsg();
  }
  export_map_t_->end();
}

} // namespace spomp
