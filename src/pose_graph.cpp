#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include "spomp/pose_graph.h"
#include "spomp/utils.h"

namespace spomp {

/**
 * @class PoseGraph
 *
 * @brief Class representing a pose graph.
 *
 * The PoseGraph class is responsible for managing the pose graph and associated operations.
 */
    PoseGraph::PoseGraph(const Params& params) : params_(params) {
  auto& tm = TimerManager::getGlobal(true);
  graph_update_t_ = tm.get("PG_graph_update");
}

/**
 * @brief Adds a new node to the pose graph.
 *
 * @param stamp The timestamp of the new node.
 * @param pose The pose of the new node.
 *
 * @return The index of the new node.
 */
    size_t PoseGraph::addNode(long stamp, const Eigen::Isometry3d& pose) {
  if (size_ > 0) {
    // Get difference from most recent pose
    const auto& most_recent_pose = pose_history_.rbegin()->second;
    Eigen::Isometry3d diff = most_recent_pose.pose.inverse() * pose;
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(most_recent_pose.key,
        P(size_), Eigen2GTSAM(diff), 
        gtsam::noiseModel::Diagonal::Sigmas(params_.between_uncertainty));

    // Use optimized version of most recent to transform to current frame
    Eigen::Isometry3d most_recent_pose_opt = 
      GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(most_recent_pose.key));
    current_opt_.insert(P(size_), Eigen2GTSAM(most_recent_pose_opt * diff));
  } else {
    initial_pose_factor_id_ = graph_.size();
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(P(size_),
        Eigen2GTSAM(Eigen::Isometry3d::Identity()),
        gtsam::noiseModel::Constrained::All(6));
    current_opt_.insert(P(size_), Eigen2GTSAM(Eigen::Isometry3d::Identity()));
  }

  pose_history_.emplace(stamp, OriginalPose{pose, P(size_)});
  ++size_;
  processGlobalBuffer();
  return size_ - 1;
}

/**
* @brief Adds a prior factor to the pose graph.
*
* This function adds a prior factor to the pose graph based on the provided timestamp and Prior2D object.
* It first stores the prior in the prior_buffer_, and then calls processGlobalBuffer() to process the buffer and add the prior factor to the appropriate node in the pose_history_.
*
* @param stamp The timestamp of the prior.
* @param prior The Prior2D object representing the prior information.
*/
    void PoseGraph::addPrior(long stamp, const Prior2D& prior) {
  prior_buffer_.emplace(stamp, prior);
  processGlobalBuffer();
}

/**
 * @brief Updates the pose graph if the conditions are met.
 *
 * This method updates the pose graph using the Levenberg-Marquardt optimization algorithm
 * every `num_frames_opt` frames, as long as the size of the graph has changed since the last
 * optimization. The optimization parameters are set to the default values.
 *
 * @note This method uses the `current_opt_` member variable to keep track of the optimized graph.
 *
 * @see gtsam::LevenbergMarquardtParams
 * @see gtsam::LevenbergMarquardtOptimizer
 */
    void PoseGraph::update() {
  if (last_opt_size_ + params_.num_frames_opt > size_ && last_opt_size_ > 0) {
    // Only optimize every num_frames_opt frames
    return;
  }

  graph_update_t_->start();
  gtsam::LevenbergMarquardtParams opt_params;
  gtsam::LevenbergMarquardtOptimizer opt(graph_, current_opt_, opt_params);
  current_opt_ = opt.optimize();
  graph_update_t_->end();

  last_opt_size_ = size_;
}

/**
 * @brief Process the global buffer by matching the priors with the pose history and adding prior factors.
 *
 * This function iterates through the prior buffer and matches each prior with the corresponding pose in the pose history.
 * If an exact match is found, the prior factor is added with the corresponding key.
 * If an exact match is not found, but there is a prior with local motion information and interpolation is allowed,
 * the prior factor is interpolated between the previous and next pose.
 * If there is no match and the priors are in the future, the process is terminated.
 *
 * @note This function modifies the prior buffer.
 */
    void PoseGraph::processGlobalBuffer() {
  for (auto prior_it = prior_buffer_.begin(); prior_it != prior_buffer_.end();) {
    // First pose after or at same time as prior
    auto matching_node_it = pose_history_.lower_bound(prior_it->first);

    // This is safe because if the first is false the second condition
    // is never checked
    if (matching_node_it != pose_history_.end() && 
        matching_node_it->first == prior_it->first) 
    {
      // Exact match
      addPriorFactor(matching_node_it->second.key, prior_it->second);
    } else if (matching_node_it != pose_history_.begin() && 
               !prior_it->second.local_pose.matrix().isIdentity(1e-5) &&
               params_.allow_interpolation) {
      // We have knowledge of local motion from keyframe
      // Get the last node with time before the prior
      --matching_node_it;
      Eigen::Isometry3d prior_T_key = prior_it->second.local_pose.inverse() * 
                                      matching_node_it->second.pose;
      
      Prior2D trans_prior = prior_it->second;
      // Current prior is world_T_prior
      trans_prior.pose = trans_prior.pose * pose32pose2(prior_T_key);
      addPriorFactor(matching_node_it->second.key, trans_prior);
    } else if (matching_node_it != pose_history_.end() && 
               params_.allow_interpolation) 
    {
      auto next_prior_it = std::next(prior_it);
      // We do not yet have a bracket
      if (next_prior_it == prior_buffer_.end()) return;

      double after_t_diff = next_prior_it->first - matching_node_it->first;
      double before_t_diff = matching_node_it->first - prior_it->first;
      
      // Make sure that we are positive
      if (after_t_diff > 0 && before_t_diff > 0) {
        const auto& after_prior = next_prior_it->second;
        const auto& before_prior = prior_it->second;

        double diff_sum = after_t_diff + before_t_diff;
        Prior2D interp_prior{};
        interp_prior.pose.translate(((after_prior.pose.translation() * before_t_diff) +
          (before_prior.pose.translation() * after_t_diff)) / diff_sum);
        interp_prior.pose.rotate(Eigen::Rotation2Dd(before_prior.pose.rotation()).slerp(
            before_t_diff/diff_sum, Eigen::Rotation2Dd(after_prior.pose.rotation())));
        // Essentially the total sigma is the magnitude of two vectors perpendicular
        if (after_prior.sigma_diag[0] > 0 && before_prior.sigma_diag[0] > 0) {
          interp_prior.sigma_diag = 
            ((after_prior.sigma_diag * before_t_diff / diff_sum).array().pow(2) +
            (before_prior.sigma_diag * after_t_diff / diff_sum).array().pow(2)).sqrt();
        }

        addPriorFactor(matching_node_it->second.key, interp_prior);
      }
    } else if (matching_node_it == pose_history_.end()) {
      // All priors in future, don't have local pose to interp
      return;
    }

    // This also advances the loop
    prior_it = prior_buffer_.erase(prior_it);
  }
}

/**
 * @brief Adds a prior factor to the pose graph.
 *
 * This function adds a prior factor to the pose graph using the given key and prior information.
 *
 * @param key The key of the pose to which the prior factor is being added.
 * @param prior The prior information used to construct the prior factor.
 */
    void PoseGraph::addPriorFactor(const gtsam::Key& key, const Prior2D& prior) {
  // (Relatively) large number
  Eigen::Vector6d unc = Eigen::Vector6d::Constant(1);
  //unc.head<2>() = Eigen::Vector2d(0.1, 0.1);
  unc[5] = 10; // z uncertainty

  // Found match
  if (graph_.exists(initial_pose_factor_id_)) {
    // Remove initial placeholder prior
    graph_.remove(initial_pose_factor_id_);
  }
  
  if (prior.sigma_diag[0] > 0) {
    // unc[2] is rot, unc[3:4] is pos
    unc.segment<3>(2) = prior.sigma_diag;
  } else {
    // sigma not specified, fall back
    unc.segment<3>(2) = params_.prior_uncertainty;
  }

  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(key,
      Eigen2GTSAM(pose22pose3(prior.pose)), 
      gtsam::noiseModel::Diagonal::Sigmas(unc));
}

/**
 * @brief Get the pose at a given time stamp.
 *
 * This function returns the pose at the specified time stamp from the pose history.
 * If no pose is found at the specified time stamp, it returns an empty std::optional value.
 *
 * @param stamp The time stamp of the pose to retrieve.
 *
 * @return The pose at the specified time stamp, or an empty std::optional if not found.
 */
    std::optional<Eigen::Isometry3d> PoseGraph::getPoseAtTime(long stamp) const {
  auto element = pose_history_.find(stamp);
  if (element == pose_history_.end()) return {};

  return GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(element->second.key));
}

/**

  * @brief Retrieve the pose from the pose graph at the specified index.
  *
  * This function returns the pose from the pose graph at the specified index.
  * If the index is out of range, an std::out_of_range exception will be thrown.
  *
  * @param ind The index of the pose to retrieve.
  * @return The pose at the specified index as an Isometry3d object.
  * @throws std::out_of_range if the index is out of range.
  */
    Eigen::Isometry3d PoseGraph::getPoseAtIndex(size_t ind) const {
  if (ind >= size_) {
    throw std::out_of_range("Index out of range of Pose Graph");
  }
  return GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(P(ind)));
}

/**
 * @brief Calculates the odom correction based on the pose history and the map pose.
 *
 * @return The odom correction as an Eigen::Isometry3d object.
 */
    Eigen::Isometry3d PoseGraph::getOdomCorrection() const {
  Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();

  if (pose_history_.size() > 0) {
    auto map_pose = getPoseAtTime(pose_history_.rbegin()->first);
    if (map_pose) {
      correction = *map_pose * pose_history_.rbegin()->second.pose.inverse();
    }
  }

  return correction;
}

} // namespace spomp
