#include "spomp/controller.h"

namespace spomp {

Controller::Controller(const Params& params) : params_(params) {}

Twistf Controller::getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
  const PanoPlanner& pano)
{
  return {};
}

std::vector<Eigen::Isometry2f> Controller::forward(
    const Eigen::Isometry2f& state, const Twistf& vel)
{
  return {};
}

float Controller::scoreTraj(const std::vector<Eigen::Isometry2f>& traj) {
  return 0;
}

bool Controller::isTrajSafe(const std::vector<Eigen::Isometry2f>& traj) {
  return false;
}

} // namespace spomp
