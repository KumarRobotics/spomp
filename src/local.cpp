#include "spomp/local.h"

namespace spomp {

Local::Local(const TerrainPano::Params& tp_p, const PanoPlanner::Params& pp_p,
    const Controller::Params& c_p) : 
  pano_(tp_p), planner_(pp_p), controller_(c_p) {}

void Local::updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose) {
  pano_.updatePano(pano, pose);
  planner_.updatePano(pano_);

  if (global_goal_.norm() != 0) {
    // Replan using new pano
    Eigen::Vector3f goal_l = pano_.getPose().inverse() * global_goal_;
    controller_.setGoal(planner_.plan(goal_l.head<2>()));
  }
}

void Local::setGoal(const Eigen::Vector3f& goal) {
  global_goal_ = goal;
  Eigen::Vector3f goal_l = pano_.getPose().inverse() * global_goal_;
  controller_.setGoal(planner_.plan(goal_l.head<2>()));
}

Twistf Local::getControlInput(const Eigen::Isometry3f& state) {
  Eigen::Isometry3f pose_l = pano_.getPose().inverse() * state;
  Eigen::Isometry2f pose_l_2 = Eigen::Isometry2f::Identity();

  // Project into 2D
  pose_l_2.translate(pose_l.translation().head<2>());
  Eigen::Vector3f rot_x = pose_l.rotation() * Eigen::Vector3f::UnitX();
  float theta = atan2(rot_x[1], rot_x[0]);
  pose_l_2.rotate(Eigen::Rotation2Df(theta));

  cur_vel_ = controller_.getControlInput(cur_vel_, pose_l_2, planner_);
  return cur_vel_;
}

} // namespace spomp
