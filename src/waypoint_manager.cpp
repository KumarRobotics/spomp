#include "spomp/waypoint_manager.h"

namespace spomp {

/**
 * @brief The WaypointManager class.
 *
 * This class represents a WaypointManager that manages waypoints based on given parameters.
 */
    WaypointManager::WaypointManager(const Params& p) : params_(p) {}

/**
 * @brief Set the path for the WaypointManager.
 *
 * This function sets the path for the WaypointManager by updating the `path_` member variable.
 * It also resets the temporary variables `next_node_` and `cur_edge_`.
 *
 * @param path A list of Node pointers representing the path.
 * @return void
 */
    void WaypointManager::setPath(const std::list<TravGraph::Node*>& path) {
  path_ = path;
  // Reset temps
  next_node_ = path_.begin();
  // We start on the "extra" edge from starting position to first node
  cur_edge_ = nullptr;
}

/**
 * @brief Sets the state of the waypoint manager based on the given position.
 *
 * This function updates the state of the waypoint manager based on the given position. It checks if there is a path
 * available and if the robot is within range of the final waypoint. If the robot has reached the final waypoint, the
 * path is cleared and the state is set to GOAL_REACHED. If the robot has reached a waypoint, the current edge is marked
 * as traversed, and the plan is advanced to the next waypoint. If neither of these conditions are met, the state is set
 * to IN_PROGRESS.
 *
 * @param pos The current position of the robot.
 * @return The state of the waypoint manager.
 *         - NO_PATH if there is no path available.
 *         - GOAL_REACHED if the robot has reached the final waypoint.
 *         - IN_PROGRESS if the robot is still following the path.
 */
    WaypointManager::WaypointState WaypointManager::setState(
    const Eigen::Vector2f& pos) 
{
  robot_pos_ = pos;
  if (path_.size() < 1) {
    // Inactive path
    return WaypointState::NO_PATH;
  }

  checkForShortcuts(pos);
  // Check if we are within threshold of final waypt
  if ((*robot_pos_ - (*path_.rbegin())->pos).norm() < params_.final_waypoint_thresh_m) {
    // We have reached the end
    path_.clear();
    return WaypointState::GOAL_REACHED;
  } else if ((*robot_pos_ - (*next_node_)->pos).norm() < params_.waypoint_thresh_m) {
    if (cur_edge_) {
      // We have traversed edge successfully, very conclusively traversable
      cur_edge_->is_locked = true;
      cur_edge_->is_experienced = true;
      cur_edge_->cls = 0;
    }
    advancePlan();
  }
  return WaypointState::IN_PROGRESS;
}

/**
 * @brief Checks for possible shortcuts in the path based on the current position.
 *
 * This function iterates through the nodes of the path starting from the next_node_ iterator.
 * It checks if the current position falls within the range of any line segment between two consecutive nodes.
 * If the position is within range and the distance squared from the line segment is below the shortcut threshold,
 * the next_node_ iterator and cur_edge_ pointer are updated.
 *
 * @param pos The current position.
 */
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

/**
 * @brief Advances the plan by incrementing the next_node_ iterator and updating the cur_edge_ variable.
 *
 * This function increments the next_node_ iterator by one. If next_node_ is not equal to the end of path_, it updates the cur_edge_ variable by retrieving the edge going to the current
* node.
 */
    void WaypointManager::advancePlan() {
  ++next_node_;
  if (next_node_ != path_.end()) {
    // Get edge going to current node
    cur_edge_ = (*next_node_)->getEdgeToNode(*std::prev(next_node_));
  }
}

/**
 * @brief Gets the next waypoint from the path.
 *
 * If the path is not empty, it returns the pointer to the next waypoint.
 * Otherwise, it returns nullptr.
 *
 * @return TravGraph::Node* The next waypoint from the path.
 */
    TravGraph::Node* WaypointManager::getNextWaypoint() const {
  if (path_.size() > 0) {
    return *next_node_;
  }
  return {};
}

/**
 * @brief Retrieves the last waypoint in the path.
 *
 * If the path contains any waypoints, this function returns a pointer to the last waypoint.
 * If the next waypoint is the beginning of the path, it returns the result of getNextWaypoint().
 * Otherwise, it returns the waypoint that comes before the next waypoint in the path.
 *
 * @return TravGraph::Node* A pointer to the last waypoint in the path. If the path is empty, nullptr is returned.
 */
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
