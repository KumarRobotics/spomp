#include "spomp/global.h"

namespace spomp {

/**
 * @class Global
 * @brief A class for initializing the Global object.
 */
    Global::Global(const Params& g_p,
                   const TravMap::Params& tm_p,
                   const TravGraph::Params& tg_p,
                   const AerialMapInfer::Params& am_p,
                   const MLPModel::Params& mlp_p,
                   const WaypointManager::Params& wm_p) :
  params_(g_p), map_(tm_p, tg_p, am_p, mlp_p), waypoint_manager_(wm_p) {}

bool Global::setGoal(const Eigen::Vector3f& goal) {
  auto pos = waypoint_manager_.getPos();
  if (!pos) {
    // Have not yet received starting location
    return false;
  }

  auto path = map_.getPath(*pos, goal.head<2>());
  if (path.size() < 1) {
    // Cannot find path
    std::cout << "\033[31m" << "[SPOMP-Global] [ERROR] Could not find valid path" 
      << "\033[0m" << std::endl;
    return false;
  }

  waypoint_manager_.setPath(path);
  return true;
}

/**
 * Updates the local reachability with the given reachability information.
 *
 * @param reachability The reachability object containing reachability information.
 * @return True if the local reachability was successfully updated, false otherwise.
 */
    bool Global::updateLocalReachability(const Reachability& reachability) {
  return updateOtherLocalReachability(reachability, 0);
}

/**
 * @brief Updates the local reachability map and performs replanning if necessary.
 *
 * This function updates the local reachability map based on the provided reachability
 * information for a specific robot. If there is an active global path, it checks the
 * current edge and updates its reachability based on the checking traversability from the
 * current position to the end of the edge. If the cost of the path after the update is
 * greater than the previous cost plus the replan hysteresis or the new cost is
 * infinity, the function performs replanning. Replanning is done by finding a new valid
 * path from the last waypoint to the end of the current path. If a valid path cannot be
 * found, the function cancels and returns false.
 *
 * @param reachability The reachability information for a robot.
 * @param robot_id The ID of the robot.
 * @return True if the update and potential replanning were successful, false otherwise.
 */
    bool Global::updateOtherLocalReachability(
    const Reachability& reachability, int robot_id) 
{
  const auto& cur_path = waypoint_manager_.getPath();
  auto cur_edge = waypoint_manager_.getCurEdge();
  auto cur_node = waypoint_manager_.getNextWaypoint();
  auto last_node = waypoint_manager_.getLastWaypoint();

  TravGraph::Edge cur_edge_copy;
  if (cur_edge) {
    cur_edge_copy = *cur_edge;
  }
  double old_cost = map_.getPathCost(cur_path);

  map_.updateLocalReachability(reachability, robot_id);

  // The rest is only relevant if there is an active global path
  // Returning true is a bit weird here, but is essentially because the plan
  // did not fail, we just do not have a plan at the moment
  if (!waypoint_manager_.havePath()) return true;

  if (cur_edge && robot_id == 0) {
    // Reload cached edge, since we want to avoid updating twice
    *cur_edge = cur_edge_copy;
    // Check this edge on the basis of checking traversability from current position
    // to the end of the edge
    map_.updateEdgeFromReachability(*cur_edge, *last_node, reachability, 
        reachability.getPose().translation());
  }
  double new_cost = map_.getPathCost(cur_path);

  if ((new_cost > old_cost + params_.replan_hysteresis ||
       new_cost == std::numeric_limits<double>::max()) && last_node) 
  {
    // We get the last waypoint because we want to replan including the current
    // edge, in case the current edge changed traversability

    // Replan
    auto new_path = map_.getPath(*last_node, *waypoint_manager_.getPath().back());
    if (new_path.size() < 1) {
      std::cout << "\033[31m" << "[SPOMP-Global] [ERROR] Could not replan to find valid path" 
        << "\033[0m" << std::endl;
      cancel();
      return false;
    } else {
      waypoint_manager_.setPath(new_path);
      waypoint_manager_.advancePlan();
      if (waypoint_manager_.getNextWaypoint() != cur_node) {
        // If this is a different node than before, 
        // then we don't want to skip the beginning
        waypoint_manager_.setPath(new_path);
      }
    }
  }

  return true;
}

} // namespace spomp
