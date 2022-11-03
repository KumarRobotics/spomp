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

void Global::updateLocalReachability(const Reachability& reachability) {
  if (!waypoint_manager_.havePath()) return;

  const auto& cur_path = waypoint_manager_.getPath();
  auto cur_edge = waypoint_manager_.getCurEdge();
  auto cur_node = waypoint_manager_.getNextWaypoint();
  auto last_node = waypoint_manager_.getLastWaypoint();

  int last_cur_edge_cls = -1;
  if (cur_edge) {
    // Cache current edge traversability
    // Want to avoid redundantly checking the current edge twice
    last_cur_edge_cls = cur_edge->cls;
  }
  float old_cost = map_.getPathCost(cur_path);

  map_.updateLocalReachability(reachability);
  if (cur_edge && cur_edge->cls == last_cur_edge_cls) {
    map_.updateEdgeFromReachability(*cur_edge, *last_node, reachability);
  }
  float new_cost = map_.getPathCost(cur_path);

  if (new_cost > old_cost) {
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
