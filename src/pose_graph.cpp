#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include "spomp/pose_graph.h"

namespace spomp {

PoseGraph::PoseGraph(const Params& params) : params_(params) {}

size_t PoseGraph::addNode(long stamp, const Eigen::Isometry3d& pose) {
  processGlobalBuffer();
  return {};
}

void PoseGraph::addPrior(long stamp, const Eigen::Isometry3d& prior) {
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

void processGlobalBuffer() {
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
