#include <queue>
#include "spomp/trav_graph.h"

namespace spomp {

TravGraph::TravGraph() {}

std::list<const TravGraph::Node*> TravGraph::getPath(
    Node* const start_n, Node* const end_n) 
{
  // Initial conditions
  reset();
  start_n->cost = 0;
  // We need to make a custom comparison, or we will end up sorting by
  // the pointers' order in mem
  auto node_comp = [](Node* a, Node* b) { return a->cost < b->cost; };
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
      Node* next_n = edge->getOtherNode(cur_n);
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
  std::list<const Node*> path;
  if (end_n->visited) {
    // We have found a path
    path.push_front(end_n);
    while (path.front() != start_n) {
      path.push_front(path.front()->best_prev_edge->getOtherNode(path.front()));
    }
  }
  return path;
}

TravGraph::Node* TravGraph::addNode(const Node& node) {
  nodes_.push_back(node);
  nodes_.back().id = nodes_.size() - 1;
  return &nodes_.back();
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
    node.visited = false;
    node.cost = std::numeric_limits<float>::infinity();
  }
}

} // namespace spomp
