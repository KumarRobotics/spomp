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

bool WaypointManager::setState(const Eigen::Vector2f& pos) {
  robot_pos_ = pos;
  if (path_.size() < 1) {
    // Inactive path
    return false;
  }

  checkForShortcuts(pos);
  // Check if we are within threshold of end
  if ((*robot_pos_ - (*next_node_)->pos).norm() < params_.waypoint_thresh_m) {
    if (cur_edge_) {
      // We have traversed edge successfully, very conclusively traversable
      cur_edge_->is_experienced = true;
      cur_edge_->cls = 0;
    }
    return advancePlan();
  }
  return false;
}

void WaypointManager::checkForShortcuts(const Eigen::Vector2f& pos) {
  const TravGraph::Node* last_node = nullptr;
  for (auto node_it=next_node_; node_it!=path_.end(); ++node_it) {
    if (!last_node) {
      last_node = *node_it;
      continue;
    }
    auto edge = (*node_it)->getEdgeToNode(last_node);
    if (!edge) continue;

    Eigen::Vector2f seg = edge->node2->pos - edge->node1->pos;
    Eigen::Vector2f pt_vec = pos - edge->node1->pos;
    float proj_mag = seg.normalized().dot(pt_vec);
    if (proj_mag >= 0 && proj_mag <= seg.norm()) {
      // We are between the line seg endpoints
      float dist_sq = std::pow(pt_vec.norm(), 2) - std::pow(proj_mag, 2);
      if (dist_sq < std::pow(params_.shortcut_thresh_m, 2)) {
        next_node_ = node_it;
        cur_edge_ = edge;
      }
    }
    last_node = *node_it;
  }
}

bool WaypointManager::advancePlan() {
  ++next_node_;
  if (next_node_ != path_.end()) {
    // Get edge going to current node
    cur_edge_ = (*next_node_)->getEdgeToNode(*std::prev(next_node_));
  } else {
    // We have reached the end
    path_.clear();
    return true;
  }
  return false;
}

TravGraph::Node* WaypointManager::getNextWaypoint() const {
  if (path_.size() > 0) {
    return *next_node_;
  }
  return {};
}

TravGraph::Node* WaypointManager::getLastWaypoint() const {
  if (path_.size() > 0) {
    if (next_node_ == path_.begin()) {
      // If the next waypoint is the beginning of the path, then there is nothing before
      return getNextWaypoint();
    } else {
      auto last_node = next_node_;
      --last_node;
      return *last_node;
    }
  }
  return {};
}

} // namespace spomp
