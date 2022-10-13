#include "spomp/global.h"

namespace spomp {

Global::Global(const TravMap::Params& tm_p, const TravGraph::Params& tg_p, 
    const WaypointManager::Params& wm_p) : 
  map_(tm_p, tg_p), waypoint_manager_(wm_p) {}

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
  auto reach_pose2 = pose32pose2(reach_pose);
  auto cur_edge = waypoint_manager_.getCurEdge();
  auto cur_node = waypoint_manager_.getNextWaypoint();
  auto last_node = waypoint_manager_.getLastWaypoint();

  bool did_change = map_.updateLocalReachability(reachability, reach_pose2);
  if (cur_edge) {
    bool did_change_cur_edge = map_.updateEdgeFromReachability(
        *cur_edge, *last_node, reachability, reach_pose2);
    did_change = did_change_cur_edge ? true : did_change;
  }

  if (did_change && waypoint_manager_.havePath()) {
    // We get the last waypoint because we want to replan including the current
    // edge, in case the current edge changed traversability
    if (last_node) {
      // Replan
      auto path = map_.getPath(*last_node, *waypoint_manager_.getPath().back());
      if (path.size() < 1) {
        // No path found
        cancel();
      } else {
        waypoint_manager_.setPath(path);
        waypoint_manager_.advancePlan();
        if (waypoint_manager_.getNextWaypoint() != cur_node) {
          // If this is a different node than before, 
          // then we don't want to skip the beginning
          waypoint_manager_.setPath(path);
        }
      }
    }
  }
}

} // namespace spomp
