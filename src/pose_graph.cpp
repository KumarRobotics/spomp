#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include "spomp/pose_graph.h"
#include "spomp/utils.h"

namespace spomp {

PoseGraph::PoseGraph(const Params& params) : params_(params) {}

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

void PoseGraph::addPrior(long stamp, const Prior2D& prior) {
  prior_buffer_.emplace(stamp, prior);
  processGlobalBuffer();
}

void PoseGraph::update() {
  if (last_opt_size_ + params_.num_frames_opt > size_ && last_opt_size_ > 0) {
    // Only optimize every num_frames_opt frames
    return;
  }

  gtsam::LevenbergMarquardtParams opt_params;
  gtsam::LevenbergMarquardtOptimizer opt(graph_, current_opt_, opt_params);
  current_opt_ = opt.optimize();

  last_opt_size_ = size_;
}

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

void PoseGraph::addPriorFactor(const gtsam::Key& key, const Prior2D& prior) {
  Eigen::Isometry3d prior_3 = Eigen::Isometry3d::Identity();
  // Large number
  Eigen::Vector6d unc = Eigen::Vector6d::Constant(100);

  // Found match
  if (graph_.exists(initial_pose_factor_id_)) {
    // Remove initial placeholder prior
    graph_.remove(initial_pose_factor_id_);
    // Initial prior should be more confident on unknown axes
    unc.setConstant(0.01);
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

std::optional<Eigen::Isometry3d> PoseGraph::getPoseAtTime(long stamp) const {
  auto element = pose_history_.find(stamp);
  if (element == pose_history_.end()) return {};

  return GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(element->second.key));
}

Eigen::Isometry3d PoseGraph::getPoseAtIndex(size_t ind) const {
  if (ind >= size_) {
    throw std::out_of_range("Index out of range of Pose Graph");
  }
  return GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(P(ind)));
}

} // namespace spomp
