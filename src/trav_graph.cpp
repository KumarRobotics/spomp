#include <queue>
#include "spomp/trav_graph.h"

namespace spomp {
// Init static members
int TravGraph::Edge::MAX_TERRAIN = 1;

/**
 * @class TravGraph
 * @brief Class representing a traversal graph
 *
 * This class provides an implementation for a traversal graph. It contains methods for
 * creating and manipulating the graph.
 */
    TravGraph::TravGraph(const Params& p) : params_(p) {
  auto& tm = TimerManager::getGlobal(true);
  get_path_t_ = tm.get("TG_get_path");
  update_edge_t_ = tm.get("TG_update_edge");
  get_near_nodes_t_ = tm.get("TG_get_near_nodes");
}

/**
 * @brief Computes the shortest path between two nodes in the graph.
 *
 * The function uses the Dijkstra algorithm to find the shortest path
 * from the start node to the end node in the graph. It considers the
 * cost of each edge and applies a custom comparison to prioritize the
 * nodes with the lowest cost. It stops as soon as it reaches the end node
 * or when there are no more nodes to visit.
 *
 * @param start_n Pointer to the start node.
 * @param end_n Pointer to the end node.
 * @return A list of nodes representing the shortest path from the
 *         start node to the end node. If there is no path, an empty
 *         list is returned.
 */
    std::list<TravGraph::Node*> TravGraph::getPath(
    Node* const start_n, Node* const end_n) 
{
  get_path_t_->start();

  // Initial conditions
  reset();
  start_n->cost = 0;
  // We need to make a custom comparison, or we will end up sorting by
  // the pointers' order in mem
  auto node_comp = [](Node* a, Node* b) { return a->cost > b->cost; };
  std::priority_queue<Node*, std::vector<Node*>, decltype(node_comp)> node_q(node_comp);
  node_q.emplace(start_n);

  // Djikstra
  while (node_q.size() > 0) {
    Node* cur_n = node_q.top();
    node_q.pop();
    if (cur_n->visited) {
      // lazy deletion
      continue;
    }
    if (cur_n == end_n) {
      // We got to the goal
      end_n->visited = true;
      break;
    }

    for (const auto& edge : cur_n->edges) {
      if (edge->cls >= Edge::MAX_TERRAIN && edge->is_locked) continue;
      Node* next_n = edge->getOtherNode(cur_n);
      if (next_n->visited) continue;
      double new_cost = cur_n->cost + edge->totalCost();
      if (new_cost < next_n->cost) {
        next_n->cost = new_cost;
        next_n->best_prev_edge = edge;
        node_q.emplace(next_n);
      }
    }
    cur_n->visited = true;
  }

  // Traverse the graph to actually find the path
  std::list<Node*> path;
  if (end_n->visited) {
    // We have found a path
    path.push_front(end_n);
    while (path.front() != start_n) {
      if (path.front()->best_prev_edge->cls >= Edge::MAX_TERRAIN &&
          path.front()->best_prev_edge->is_locked) 
      {
        // We have a bad edge
        path = {};
        break;
      }
      path.push_front(path.front()->best_prev_edge->getOtherNode(path.front()));
    }
  }

  get_path_t_->end();
  return path;
}

/**
* @brief Verifies if a given node has a valid edge to exit from.
*
* If a node does not have a valid edge to exit from, this function unlocks all the edges of the node.
*
* @param node Pointer to the node to be verified
*/
    void TravGraph::verifyCanExitNode(Node* const node) {
  // Make sure that it is possible to exit this node.
  // We do this so that robot can't get stuck with no valid edges

  for (const auto& edge : node->edges) {
    if (edge->cls < Edge::MAX_TERRAIN || !edge->is_locked) {
      // Valid edge out of here
      return;
    }
  }

  for (auto& edge : node->edges) {
    edge->is_locked = false;
  }
}

/**
 * @brief Calculates the cost of a given path in the Traveler's Graph.
 *
 * This function calculates the cost of traversing a given path in the Traversability Graph.
 * The cost is calculated by summing up the total cost of each edge in the path.
 * If an edge is locked or has a terrain class greater than or equal to MAX_TERRAIN, the function will return infinity.
 * If the path is empty, the cost will be zero.
 *
 * @param path A list of Node pointers representing the path to calculate the cost for.
 * @return The cost of traversing the given path in the Traveler's Graph.
 * @return If the path is invalid or contains locked or impassable edges, std::numeric_limits<double>::max() will be returned.
 */
    double TravGraph::getPathCost(const std::list<Node*>& path) const {
  double cost = 0;

  const Node* last_node = nullptr;
  for (const auto& node : path) {
    if (!last_node || last_node == node) {
      last_node = node;
      continue;
    }
    auto edge = node->getEdgeToNode(last_node);
    if (!edge || (edge->cls >= Edge::MAX_TERRAIN && edge->is_locked)) {
      return std::numeric_limits<double>::max();
    }
    cost += node->getEdgeToNode(last_node)->totalCost();
    last_node = node;
  }

  return cost;
}

/**
 * @brief Calculates the length of a path in a graph.
 *
 * This function takes a list of nodes that represent a path in a graph and
 * calculates the total length of the path. The length is computed by summing
 * the lengths of the edges between consecutive nodes in the list.
 *
 * If an edge is not found between two adjacent nodes, the function returns
 * std::numeric_limits<float>::max() to indicate an invalid path.
 *
 * @param path The list of nodes representing the path.
 * @return The length of the path, or std::numeric_limits<float>::max() if an
 *         invalid path is given.
 */
    float TravGraph::getPathLength(const std::list<Node*>& path) const {
  float length = 0;

  const Node* last_node = nullptr;
  for (const auto& node : path) {
    if (!last_node || last_node == node) {
      last_node = node;
      continue;
    }
    auto edge = node->getEdgeToNode(last_node);
    if (edge) {
      length += edge->length;
    } else {
      return std::numeric_limits<float>::max();
    }
    last_node = node;
  }

  return length;
}

/**
 * @brief Updates the local reachability based on the given reachability object.
 *
 * @param reachability The reachability object containing information about the reachability.
 * @return True if the map has changed, false otherwise.
 */
    bool TravGraph::updateLocalReachability(const Reachability& reachability)
{
  if (edges_.empty()) {
    return false;
  }

  auto near_nodes = getNodesNear(reachability.getPose().translation(), 
      params_.reach_node_max_dist_m);

  bool did_map_change = false;
  for (const auto& node_ptr : near_nodes) {
    for (const auto& edge : node_ptr->edges) {
      if (updateEdgeFromReachability(*edge, *node_ptr, reachability)) {
        did_map_change = true;
      }
    }
  }

  return did_map_change;
}

/**
 * @brief Update an edge based on reachability information.
 *
 * This function updates an edge based on reachability information. It analyzes the edge
 * based on the start position, end position, and provided reachability analysis parameters.
 * Depending on the reachability analysis result, the edge's properties may be modified.
 *
 * @param edge The edge to update.
 * @param start_node The start node of the edge.
 * @param reachability The reachability information.
 * @param start_pos The optional start position.
 * @return True if the map changed due to the edge update, false otherwise.
 */
    bool TravGraph::updateEdgeFromReachability(TravGraph::Edge& edge,
                                               const TravGraph::Node& start_node, const Reachability& reachability,
                                               std::optional<Eigen::Vector2f> start_pos)
{
  update_edge_t_->start();

  TravGraph::Node* dest_node_ptr = edge.getOtherNode(&start_node);
  Reachability::EdgeExperience edge_exp;
  if (start_pos) {
    edge_exp = reachability.analyzeEdge(*start_pos, dest_node_ptr->pos,
        {params_.trav_window_rad, params_.max_trav_discontinuity_m});
  } else {
    edge_exp = reachability.analyzeEdge(start_node.pos, dest_node_ptr->pos,
        {params_.trav_window_rad, params_.max_trav_discontinuity_m});
  }

  bool did_map_change = false;
  if (edge_exp == Reachability::TRAV) {
    if (edge.is_locked) {
      edge.untrav_counter = -params_.num_edge_exp_before_mark;
    } else {
      did_map_change = true;
      edge.decUntravCounter();
      if (edge.untrav_counter <= -params_.num_edge_exp_before_mark) {
        // Requires multiple markings in a row to be locked in
        // Only lock in if was experienced by this robot
        edge.is_locked = !reachability.isOtherRobot();
      }
      edge.cls = 0;
      // Don't want 0 cost so length still matters
      // Just want very small number
      edge.cost = -std::log(params_.trav_edge_prob_trav);
    }
  } else if (edge_exp == Reachability::NOT_TRAV) {
    // Want to be able to mark untrav even if marked as experienced
    if (edge.cls == Edge::MAX_TERRAIN) {
      // Requires two markings in a row to be locked in
      edge.is_locked = !reachability.isOtherRobot();
    }
    edge.incUntravCounter();
    if (edge.untrav_counter >= params_.num_edge_exp_before_mark) {
      // enough strikes that you're out
      did_map_change = true;

      // almost unreachable cost
      edge.cls = Edge::MAX_TERRAIN-1;
      if (!reachability.isOtherRobot()) {
        // Unreachable cost
        ++edge.cls;
      }
    }
  } else if (edge_exp == Reachability::UNKNOWN && !edge.is_locked) {
    edge.untrav_counter = 0;
  }

  update_edge_t_->end();
  return did_map_change;
}

/**
 * @brief Retrieve nodes near a specified position within a given delta.
 *
 * This function returns a vector of pointers to the nodes that are located
 * near the specified position within the specified delta. The position is
 * provided as a 2D vector (Eigen::Vector2f), and the delta represents the maximum
 * distance from the position for a node to be considered "near". Nodes are
 * considered "near" if the distance between their positions and the specified
 * position is less than or equal to the delta.
 *
 * @param pos   The position to search around.
 * @param delta The maximum distance from the position to consider a node "near".
 *
 * @return A vector of pointers to the nodes that are near the specified position.
 */
    const std::vector<TravGraph::Node*> TravGraph::getNodesNear(
    const Eigen::Vector2f& pos, float delta) 
{
  get_near_nodes_t_->start();
  std::vector<TravGraph::Node*> near_nodes{};
  for (auto& [node_id, node] : nodes_) {
    if ((node.pos - pos).norm() <= delta) {
      near_nodes.push_back(&node);
    }
  }
  get_near_nodes_t_->end();
  return near_nodes;
}

/**
 * @brief Adds a new node to the graph.
 *
 * This function creates a new node with the provided information and
 * adds it to the graph.
 *
 * @param node The node to be added.
 * @return A pointer to the added node.
 */
    TravGraph::Node* TravGraph::addNode(const Node& node) {
  int id = nodes_.size();
  auto [it, success] = nodes_.emplace(id, node);
  it->second.id = id;
  return &(it->second);
}

/**
 * @brief Adds an edge to the graph.
 *
 * This function verifies that the endpoints of the edge are valid and that the edge does not already exist
 * in the graph. If the conditions are met, the edge is added to the graph's list of edges and also assigned
 * to the nodes it connects.
 *
 * @param edge The edge to be added to the graph.
 * @return A pointer to the added edge if it was successfully added, nullptr otherwise.
 */
    TravGraph::Edge* TravGraph::addEdge(const Edge& edge) {
  // Verify that endpoints are valid and edge does not already exist
  if (edge.node1 && edge.node2 && 
      !(edge.node1->getEdgeToNode(edge.node2))) 
  {
    edges_.push_back(edge);

    // Add edge to nodes
    Edge* edge_ptr = &edges_.back();
    edge.node1->edges.push_back(edge_ptr);
    edge.node2->edges.push_back(edge_ptr);
    return edge_ptr;
  }
  return nullptr;
}

/**
 * @brief Resets the visited flag and cost of all the nodes in the graph.
 *
 * This function iterates over all the nodes in the `nodes_` map and resets their visited flag
 * and cost. The visited flag is set to `false` and the cost is set to positive infinity.
 *
 * Complexity:
 *  - Time: O(N), where N is the number of nodes in the graph.
 *  - Space: O(1)
 *
 * Example usage:
 *
 * ```cpp
 * TravGraph graph;
 * // ... add nodes and edges to the graph
 * graph.reset();
 * ```
 */
    void TravGraph::reset() {
  for (auto& node : nodes_) {
    node.second.visited = false;
    node.second.cost = std::numeric_limits<double>::infinity();
  }
}

} // namespace spomp
