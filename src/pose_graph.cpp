#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include "spomp/pose_graph.h"

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
    if (pose_history_.size() > 0) {
      if (prior_it->first > pose_history_.rbegin()->first) {
        // All priors are in the future, leave in the buffer for now
        // Assumption here that poses arrive in order
        return;
      }
    }

    auto matching_node = pose_history_.find(prior_it->first);
    if (matching_node != pose_history_.end()) {
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
      
      if (prior_it->second.sigma_diag[0] > 0) {
        // unc[2] is rot, unc[3:4] is pos
        unc.segment<3>(2) = prior_it->second.sigma_diag;
      } else {
        // sigma not specified, fall back
        unc.segment<3>(2) = params_.prior_uncertainty;
      }
      // Convert 2D pose to 3D
      prior_3.translation().head<2>() = prior_it->second.pose.translation();
      prior_3.rotate(Eigen::AngleAxisd(
            Eigen::Rotation2Dd(prior_it->second.pose.rotation()).angle(), 
                               Eigen::Vector3d::UnitZ()));

      graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(matching_node->second.key,
          Eigen2GTSAM(prior_3), 
          gtsam::noiseModel::Diagonal::Sigmas(unc));
    }
    prior_it = prior_buffer_.erase(prior_it);
  }
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
