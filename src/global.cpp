#include "spomp/global.h"

namespace spomp {

Global::Global(const TravMap::Params& tm_p, const WaypointManager::Params& wm_p) : 
  map_(tm_p), waypoint_manager_(wm_p) {}

bool Global::setGoal(const Eigen::Vector3f& goal) {
  auto pos = waypoint_manager_.getPos();
  if (!pos) {
    // Have not yet received starting location
    return false;
  }

  auto path = map_.getPath(*pos, goal.head<2>());
  if (path.size() < 1) {
    // Cannot find path
    return false;
  }

  waypoint_manager_.setPath(path);
  return true;
}

void Global::updateLocalReachability(const Reachability& reachability, 
    const Eigen::Isometry3f& reach_pose)
{
  bool did_change = map_.updateLocalReachability(reachability, pose32pose2(reach_pose));

  if (did_change && waypoint_manager_.havePath()) {
    // We get the last waypoint because we want to replan including the current
    // edge, in case the current edge changed traversability
    auto pos = waypoint_manager_.getLastWaypoint();
    if (pos) {
      // Replan
      auto path = map_.getPath(*pos, waypoint_manager_.getPath().back()->pos);
      if (path.size() < 1) {
        // No path found
        waypoint_manager_.cancel();
      } else {
        waypoint_manager_.setPath(path);
        waypoint_manager_.advancePlan();
      }
    }
  }
}

} // namespace spomp
