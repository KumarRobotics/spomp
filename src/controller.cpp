#include "spomp/controller.h"

namespace spomp {

Controller::Controller(const Params& params) : params_(params) {}

Twistf Controller::getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
    const Eigen::Vector2f& goal, const PanoPlanner& planner) const
{
  Twistf max_delta(params_.max_lin_accel / params_.freq, 
                   params_.max_ang_accel / params_.freq);
  Twistf max_twist = cur_vel + max_delta;
  Twistf min_twist = cur_vel - max_delta;

  // Sample the control space, obeying the bounds
  Eigen::VectorXf lin_samples = Eigen::VectorXf::LinSpaced(
      params_.lin_disc, 
      std::clamp<float>(min_twist.linear(), 0, params_.max_lin_vel), 
      std::clamp<float>(max_twist.linear(), 0, params_.max_lin_vel));
  Eigen::VectorXf ang_samples = Eigen::VectorXf::LinSpaced(
      params_.ang_disc, 
      std::clamp<float>(min_twist.ang(), -params_.max_ang_vel, params_.max_ang_vel), 
      std::clamp<float>(max_twist.ang(), -params_.max_ang_vel, params_.max_ang_vel));

  // Forward simulate all the trajectories and pick the safest one with lowest cost
  Twistf best_twist{};
  float best_cost = std::numeric_limits<float>::infinity();
  std::vector<Eigen::Isometry2f> traj;
  for (int lin_i=0; lin_i<lin_samples.size(); ++lin_i) {
    for (int ang_i=0; ang_i<ang_samples.size(); ++ang_i) {
      Twistf t(lin_samples[lin_i], ang_samples[ang_i]);
      traj = forward(state, t);
      if (isTrajSafe(traj, planner)) {
        float cost = trajCost(traj, goal);
        if (cost < best_cost) {
          best_cost = cost;
          best_twist = t;
        }
      }
    }
  }

  return best_twist;
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

float Controller::trajCost(const std::vector<Eigen::Isometry2f>& traj,
    const Eigen::Vector2f& goal) const {
  if (traj.size() < 1) {
    return -1;
  }

  // We don't need a velocity penalty, because a faster velocity
  // will mean we get to goal faster, so lin_dist will be smaller
  float lin_dist = (traj.back().translation() - goal).norm();
  // Angular cost to allow robot to have reason to turn in place
  float ang_dist = angularDist(traj.back(), goal);
  // Normally regAngle has range [0, 2pi), we want [-pi, pi)
  ang_dist = abs(ang_dist - pi);

  // ang_dist matters less if we are closer to the goal
  return lin_dist + 0.1 * lin_dist * ang_dist;
}

float Controller::angularDist(const Eigen::Isometry2f& pose,
    const Eigen::Vector2f& goal)
{
  Eigen::Vector2f goal_vec = pose.translation() - goal;
  float goal_dir = atan2(goal_vec[1], goal_vec[0]);
  float pose_dir = Eigen::Rotation2Df(pose.rotation()).angle();
  return regAngle(goal_dir - pose_dir);
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
