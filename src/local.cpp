#include "spomp/local.h"

namespace spomp {

Local::Local(const Local::Params& l_p, const TerrainPano::Params& tp_p, 
    const PanoPlanner::Params& pp_p, const Controller::Params& c_p) : 
  params_(l_p), pano_(tp_p), planner_(pp_p), controller_(c_p) {}

/**
 * @brief Updates the panorama, pose, and global goal of the Local object.
 *
 * This function updates the panorama, pose, and global goal of the Local object.
 *
 * @param pano The panorama data as an Eigen 2D array of floats.
 * @param pose The pose of the Local object as an Eigen Isometry3f.
 * @param global_goal The global goal as an Eigen Vector3f.
 */
    void Local::updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose,
                           const Eigen::Vector3f& global_goal)
{
  if (global_goal.norm() > 0 && global_goal_) {
    // We have an updated version of the current goal
    global_goal_ = global_goal;
  }
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

/**

     * @brief Set the goal for the local planner.

     * This function takes in a 3D vector representing the goal position
     * in the global coordinate system and sets it as the goal for the
     * local planner. It performs necessary transformations to convert
     * the global goal position to the local coordinate system of the robot.

     * @param[in] goal A 3D vector representing the goal position in the global coordinate system.

     * @return None.

     */
    void Local::setGoal(const Eigen::Vector3f& goal) {
  global_goal_ = goal;
  Eigen::Vector3f goal_l = pano_.getPose().inverse() * *global_goal_;
  controller_.setGoal(planner_.plan(goal_l.head<2>()));
}

/**
 * @brief Get the control input based on the current state.
 *
 * This function calculates the control input based on the given state.
 * If a global goal is set and the current position is close enough to the global goal,
 * it stops the motion by resetting the global goal and setting the current velocity to zero.
 * Otherwise, it checks if the current pose is close enough to the local goal,
 * and if so, it replans the motion by setting a new global goal based on the previous one.
 * Finally, it calculates the control input based on the current velocity,
 * the pose in local frame, the planner, and the panorama.
 *
 * @param state The current state of the system represented by an Isometry3f.
 * @return The control input represented by a Twistf.
 */
    Twistf Local::getControlInput(const Eigen::Isometry3f& state) {
  if (global_goal_) {
    if ((*global_goal_ - state.translation()).norm() < 
        params_.global_goal_thresh_m) 
    {
      // Reached global goal, stop
      global_goal_.reset();
      cur_vel_ = {0, 0};
    } else {
      Eigen::Isometry3f pose_l = pano_.getPose().inverse() * state;
      if ((controller_.getLocalGoal() - pose_l.translation().head<2>()).norm() < 
          params_.local_goal_thresh_m) 
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
