#include "spomp/global.h"

namespace spomp {

Global::Global(const TravMap::Params& tm_p, const TravGraph::Params& tg_p, 
    const AerialMapInfer::Params& am_p, const MLPModel::Params& mlp_p,
    const WaypointManager::Params& wm_p) : 
  map_(tm_p, tg_p, am_p, mlp_p), waypoint_manager_(wm_p) {}

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

void Global::updateLocalReachability(const Reachability& reachability) {
  updateOtherLocalReachability(reachability, 0);
}

void Global::updateOtherLocalReachability(
    const Reachability& reachability, int robot_id) 
{
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

  map_.updateLocalReachability(reachability, robot_id);

  // The rest is only relevant if there is an active global path
  if (!waypoint_manager_.havePath()) return;

  if (cur_edge && cur_edge->cls == last_cur_edge_cls) {
    // Check this edge on the basis of checking traversability from current position
    // to the end of the edge
    map_.updateEdgeFromReachability(*cur_edge, *last_node, reachability, 
        reachability.getPose().translation());
  }
  float new_cost = map_.getPathCost(cur_path);

  if ((new_cost > old_cost + 2 || 
       new_cost > std::pow(1000, TravGraph::Edge::MAX_TERRAIN)-1) && 
      last_node) 
  {
    // We get the last waypoint because we want to replan including the current
    // edge, in case the current edge changed traversability

    // Replan
    auto new_path = map_.getPath(*last_node, *waypoint_manager_.getPath().back());
    if (new_path.size() < 1) {
      std::cout << "\033[31m" << "[SPOMP-Global] Attemping to reset local graph" 
        << "\033[0m" << std::endl;
      if (waypoint_manager_.getPos()) {
        map_.resetGraphAroundPoint(*waypoint_manager_.getPos());
        new_path = map_.getPath(*last_node, *waypoint_manager_.getPath().back());
      }

      if (new_path.size() < 1) {
        // Cannot find path
        std::cout << "\033[31m" << "[SPOMP-Global] [ERROR] Could not find valid path" 
          << "\033[0m" << std::endl;
        cancel();
      } else {
        waypoint_manager_.setPath(new_path);
      }
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
}

} // namespace spomp
