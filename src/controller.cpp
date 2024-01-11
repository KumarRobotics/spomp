#include "spomp/controller.h"

#include <iostream>

namespace spomp {

/**
 * @class Controller
 * @brief This class represents a controller object
 *
 * @details The Controller class is responsible for controlling a system using the provided parameters.
 * It utilizes a TimerManager for timing purposes.
 *
 * @param params The parameters needed for controller initialization
 */
    Controller::Controller(const Params& params) : params_(params) {
  auto& tm = TimerManager::getGlobal();
  controller_t_ = tm.get("CO");
}

/**
 * @brief Calculates the control input for the robot given the current velocity, robot state,
 *        planner, and terrain panorama.
 *
 * @param cur_vel       The current velocity of the robot.
 * @param state_p       The state of the robot as an isometry transformation.
 * @param planner       The PanoPlanner used for planning.
 * @param pano          The TerrainPano used for traversability information.
 * @return              The control input (Twistf) for the robot.
 */
    Twistf Controller::getControlInput(const Twistf& cur_vel, const Eigen::Isometry3f& state_p,
                                       const PanoPlanner& planner, const TerrainPano& pano) const
{
  controller_t_->start();

  // Transform into the control space
  Eigen::Isometry3f state_3 = state_p * params_.control_trans;
  Eigen::Isometry2f state = pose32pose2(state_3);
  
  // Compute delta bounds
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
      // Require that robot always be moving
      // This helps robot not get stuck in local min
      if (t.linear() < max_delta.linear()*0.8 && t.ang() < max_delta.ang()*0.8) continue;
      traj = forward(state, t);

      float obs_cost = trajCostObs(traj, pano);
      float goal_cost = trajCostGoal(traj);
      float cost = goal_cost + params_.obs_cost_weight * obs_cost;
      if (!isTrajSafe(traj, planner)) {
        // If not safe, cost is really high, so essentially only relevant
        // if there are no safe options.  If so, priority is just getting
        // to safety
        cost = 100 + 10*obs_cost + trajCostGoal(traj);
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_twist = t;
      }
    }
  }

  controller_t_->end();
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

/**
 * Calculates the cost of a trajectory with respect to a goal.
 * The cost is the sum of the linear distance from the point slightly in front of the robot to the goal,
 * and the cross product of the vector from the point to the goal with the goal vector.
 * This encourages the robot to face forward and stay on the path towards the goal.
 *
 * @param traj The trajectory as a vector of Isometry2f
 *
 * @return The cost of the trajectory with respect to the goal.
 * If the trajectory is empty, -1 is returned.
 */
    float Controller::trajCostGoal(const std::vector<Eigen::Isometry2f>& traj) const {
  if (traj.size() < 1) {
    return -1;
  }

  // Get point slightly in front of robot
  // This means that rotation affects cost and encourages facing forward
  Eigen::Vector2f pt_front = traj.back() * (Eigen::Vector2f::UnitX()*0.5);

  // We don't need a velocity penalty, because a faster velocity
  // will mean we get to goal faster, so lin_dist will be smaller
  float lin_dist = (pt_front - goal_).norm();
  // Path is just line from origin to goal, find distance with cross prod
  float path_dist = crossNorm(pt_front, goal_.normalized());
  if (path_dist < 1) {
    // We are essentially on the path, and this cost can have the
    // undesirable effect of making the robot travel backwards on the path
    path_dist = 0;
  }

  return lin_dist + path_dist;
}

/**
 * Calculates the cost of traversing a trajectory with obstacles
 *
 * @param traj The trajectory as a vector of 2D Isometry transformation matrices
 * @param pano The TerrainPano object representing the terrain and obstacles
 * @return The cost of traversing the trajectory
 */
    float Controller::trajCostObs(const std::vector<Eigen::Isometry2f>& traj,
                                  const TerrainPano& pano) const {
  float cost = 0;
  for (const auto& pt : traj) {
    Eigen::Vector2f pt_front = pt * (Eigen::Vector2f::UnitX()*0.5);
    cost -= pano.getObstacleDistAt(pt_front);
  }
  return cost/traj.size();
}

/**
 * @brief Calculates the angular distance between a given pose and a goal point.
 *
 * The angular distance is defined as the difference in orientation angles between
 * the given pose and the line that connects the pose's translation with the goal point.
 *
 * @param pose   The pose at which the angular distance is calculated.
 * @param goal   The goal point to which the angular distance is measured.
 * @return       The angular distance between the pose and the goal point.
 */
    float Controller::angularDist(const Eigen::Isometry2f& pose,
                                  const Eigen::Vector2f& goal)
{
  Eigen::Vector2f goal_vec = pose.translation() - goal;
  float goal_dir = atan2(goal_vec[1], goal_vec[0]);
  float pose_dir = Eigen::Rotation2Df(pose.rotation()).angle();
  return regAngle(goal_dir - pose_dir);
}

/**
 * @brief Checks if a given trajectory is safe.
 *
 * This method checks if every point in the trajectory is safe based on the planner's reachability.
 *
 * @param traj A vector of 2D isometric transformations representing the trajectory.
 * @param planner The PanoPlanner instance used to check if a point is safe.
 * @return True if every point in the trajectory is safe, false otherwise.
 */
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
