#include <queue>
#include "spomp/trav_graph.h"

namespace spomp {
// Init static members
int TravGraph::Edge::MAX_TERRAIN = 1;

TravGraph::TravGraph(const Params& p) : params_(p) {
  auto& tm = TimerManager::getGlobal();
  get_path_t_ = tm.get("TG_get_path");
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
      if (edge->is_experienced && edge->cls > Edge::MAX_TERRAIN) continue;
      Node* next_n = edge->getOtherNode(cur_n);
      if (next_n->visited) continue;
      float new_cost = cur_n->cost + edge->totalCost();
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
      path.push_front(path.front()->best_prev_edge->getOtherNode(path.front()));
    }
  }

  get_path_t_->end();
  return path;
}

bool TravGraph::updateLocalReachability(const Reachability& reachability, 
    const Eigen::Isometry2f& reach_pose)
{
  if (edges_.empty()) {
    return false;
  }

  auto near_nodes = getNodesNear(reach_pose.translation(), 
      params_.reach_node_max_dist_m);

  bool did_map_change = false;
  for (const auto& node_ptr : near_nodes) {
    for (const auto& edge : node_ptr->edges) {
      if (updateEdgeFromReachability(*edge, *node_ptr, reachability, reach_pose)) {
        did_map_change = true;
      }
    }
  }

  return did_map_change;
}

bool TravGraph::updateEdgeFromReachability(TravGraph::Edge& edge, 
    const TravGraph::Node& start_node, const Reachability& reachability,
    const Eigen::Isometry2f& reach_pose)
{
  TravGraph::Node* dest_node_ptr = edge.getOtherNode(&start_node);
  Eigen::Vector2f local_dest_pose = reach_pose.inverse() * dest_node_ptr->pos;

  float range = local_dest_pose.norm();
  float bearing = atan2(local_dest_pose[1], local_dest_pose[0]);

  bool not_reachable = true;
  bool reachable = true;
  for (float b=bearing-params_.trav_window_rad; b<=bearing+params_.trav_window_rad; 
       b+=std::abs(reachability.proj.delta_angle)) 
  {
    int ind = reachability.proj.indAt(b);
    if (range <= reachability.scan[ind] || !reachability.is_obs[ind]) {
      // We have a non-obstacle path
      not_reachable = false;
    }
    if (range > reachability.scan[ind]) {
      // We have an obstacle or unknown path
      reachable = false;
      if (reachability.scan[ind] > params_.reach_max_dist_to_be_obs_m) {
        // We think we can't get there, but it is far away.  Unclear.
        not_reachable = false;
      }
    }
  }

  bool did_map_change = false;
  if (!edge.is_experienced) {
    if (not_reachable) {
      // Unreachable cost
      if (edge.cls != Edge::MAX_TERRAIN + 1) {
        did_map_change = true;
      } else {
        // Requires two markings in a row to be locked in
        edge.is_experienced = true;
      }
      edge.cls = Edge::MAX_TERRAIN + 1;
    } else if (reachable) {
      if (edge.cls != 0) {
        did_map_change = true;
      } else {
        // Requires two markings in a row to be locked in
        edge.is_experienced = true;
      }
      edge.cls = 0;
    }
  }

  return did_map_change;
}

const std::vector<TravGraph::Node*> TravGraph::getNodesNear(
    const Eigen::Vector2f& pos, float delta) 
{
  std::vector<TravGraph::Node*> near_nodes{};
  for (auto& [node_id, node] : nodes_) {
    if ((node.pos - pos).norm() <= delta) {
      near_nodes.push_back(&node);
    }
  }
  return near_nodes;
}

TravGraph::Node* TravGraph::addNode(const Node& node) {
  int id = nodes_.size();
  auto [it, success] = nodes_.emplace(id, node);
  it->second.id = id;
  return &(it->second);
}

void TravGraph::addEdge(const Edge& edge) {
  if (edge.node1 && edge.node2) {
    edges_.push_back(edge);

    // Add edge to nodes
    Edge* edge_ptr = &edges_.back();
    edge.node1->edges.push_back(edge_ptr);
    edge.node2->edges.push_back(edge_ptr);
  }
}

void TravGraph::reset() {
  for (auto& node : nodes_) {
    node.second.visited = false;
    node.second.cost = std::numeric_limits<float>::infinity();
  }
}

} // namespace spomp
