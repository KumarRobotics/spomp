#include "spomp/controller.h"

namespace spomp {

Controller::Controller(const Params& params) : params_(params) {}

Twistf Controller::getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
    const PanoPlanner& planner) const
{
  return {};
}

std::vector<Eigen::Isometry2f> Controller::forward(
    const Eigen::Isometry2f& state, const Twistf& vel) const
{
  float theta = Eigen::Rotation2Df(state.rotation()).angle();

  Eigen::Isometry2f delta_state = Eigen::Isometry2f::Identity();
  // Avoid numerical instabilities for very small omega
  if (std::abs(vel.ang()) < 1e-5) {
    // Just drive forward at the given speed
    delta_state.translate(params_.horizon_dt * vel.linear() * 
        Eigen::Vector2f(cos(theta), sin(theta)));
  } else {
    float R = vel.linear() / vel.ang();
    Eigen::Vector2f ICC = state.translation() + 
      R * Eigen::Vector2f(-sin(theta), cos(theta));
    Eigen::Rotation2Df delta_rot(params_.horizon_dt * vel.ang());
    delta_state.translate(ICC - delta_rot * ICC);
    delta_state.rotate(delta_rot);
  }

  std::vector<Eigen::Isometry2f> traj;
  traj.reserve(std::ceil(params_.horizon_sec / params_.horizon_dt));
  traj.push_back(state);
  for (float t=0; t<=params_.horizon_sec; t+=params_.horizon_dt) {
    traj.push_back(delta_state * traj.back());
  }
  return traj;
}

float Controller::scoreTraj(const std::vector<Eigen::Isometry2f>& traj) const {
  return 0;
}

bool Controller::isTrajSafe(const std::vector<Eigen::Isometry2f>& traj,
    const PanoPlanner& planner) const 
{
  for (const auto& pt : traj) {
    if (!planner.isSafe(pt.translation())) {
      return false;
    }
  }
  return true;
}

} // namespace spomp
