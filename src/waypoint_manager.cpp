#include "spomp/waypoint_manager.h"

namespace spomp {

WaypointManager::WaypointManager(const Params& p) : params_(p) {}

void WaypointManager::setPath(const std::list<TravGraph::Node*>& path) {
  path_ = path;
  // Reset temps
  next_node_ = path_.begin();
  // We start on the "extra" edge from starting position to first node
  cur_edge_ = nullptr;
}

void WaypointManager::setState(const Eigen::Vector2f& pos) {
  robot_pos_ = pos;
  if (path_.size() < 1) {
    // Inactive path
    return;
  }

  if ((*robot_pos_ - (*next_node_)->pos).norm() < params_.waypoint_thresh_m) {
    ++next_node_;
    if (next_node_ != path_.end()) {
      // Get edge going to current node
      cur_edge_ = (*next_node_)->getEdgeToNode(*std::prev(next_node_));
    } else {
      // We have reached the end
      path_.clear();
    }
  }
}

std::optional<Eigen::Vector2f> WaypointManager::getNextWaypoint() {
  if (path_.size() > 0) {
    return (*next_node_)->pos;
  }
  return {};
}

} // namespace spomp
