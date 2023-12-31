#include <queue>
#include "spomp/trav_graph.h"

namespace spomp {
// Init static members
int TravGraph::Edge::MAX_TERRAIN = 1;

TravGraph::TravGraph(const Params& p) : params_(p) {
  auto& tm = TimerManager::getGlobal(true);
  get_path_t_ = tm.get("TG_get_path");
  update_edge_t_ = tm.get("TG_update_edge");
  get_near_nodes_t_ = tm.get("TG_get_near_nodes");
}

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

TravGraph::Node* TravGraph::addNode(const Node& node) {
  int id = nodes_.size();
  auto [it, success] = nodes_.emplace(id, node);
  it->second.id = id;
  return &(it->second);
}

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

void TravGraph::reset() {
  for (auto& node : nodes_) {
    node.second.visited = false;
    node.second.cost = std::numeric_limits<double>::infinity();
  }
}

} // namespace spomp
