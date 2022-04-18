#include "spomp/local.h"

namespace spomp {

Local::Local(const Local::Params& l_p, const TerrainPano::Params& tp_p, 
    const PanoPlanner::Params& pp_p, const Controller::Params& c_p) : 
  params_(l_p), pano_(tp_p), planner_(pp_p), controller_(c_p) {}

void Local::updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose) {
  Eigen::Vector3f last_local_l = Eigen::Vector3f::Zero();
  last_local_l.head<2>() = controller_.getLocalGoal();
  Eigen::Vector3f old_goal_g = pano_.getPose() * last_local_l;

  pano_.updatePano(pano, pose);
  planner_.updatePano(pano_);

  if (global_goal_) {
    // Replan using new pano
    Eigen::Vector3f goal_l = pano_.getPose().inverse() * *global_goal_;
    Eigen::Vector3f old_goal_l = Eigen::Vector3f::Zero(); 
    if (last_local_l.norm() != 0) {
      old_goal_l = pano_.getPose().inverse() * old_goal_g;
    }
    controller_.setGoal(planner_.plan(goal_l.head<2>(), old_goal_l.head<2>()));
  }
}

void Local::setGoal(const Eigen::Vector3f& goal) {
  global_goal_ = goal;
  Eigen::Vector3f goal_l = pano_.getPose().inverse() * *global_goal_;
  controller_.setGoal(planner_.plan(goal_l.head<2>()));
}

Twistf Local::getControlInput(const Eigen::Isometry3f& state) {
  if (global_goal_) {
    if ((*global_goal_ - state.translation()).norm() < 
        params_.goal_thresh_m) 
    {
      // Reached global goal, stop
      global_goal_.reset();
      cur_vel_ = {0, 0};
    } else {
      Eigen::Isometry3f pose_l = pano_.getPose().inverse() * state;
      if ((controller_.getLocalGoal() - pose_l.translation().head<2>()).norm() < 
          params_.goal_thresh_m) 
      {
        // Close to local goal, replan
        setGoal(*global_goal_);
      }
      cur_vel_ = controller_.getControlInput(cur_vel_, pose_l, planner_, pano_);
    }
  } else {
    cur_vel_ = {0, 0};
  }

  return cur_vel_;
}

} // namespace spomp
