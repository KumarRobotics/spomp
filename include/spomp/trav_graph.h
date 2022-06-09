#pragma once

#include <iostream>
#include <map>
#include <list>
#include <Eigen/Dense>
#include "spomp/timer.h"

namespace spomp {

/*!
 * Traversability graph representation
 */
class TravGraph {
  public:
    TravGraph();

    // Forward declaration
    struct Edge;
    struct Node {
      int id{0};
      std::vector<Edge*> edges{};
      Eigen::Vector2f pos{0, 0};

      // Graph-search temps
      bool visited{false};
      float cost{std::numeric_limits<float>::infinity()};
      Edge* best_prev_edge{nullptr};

      Node(const Eigen::Vector2f& p) : pos(p) {}
      Node() = default;

      Edge* getEdgeToNode(const Node* n) {
        for (const auto& edge : edges) {
          if (edge->getOtherNode(this) == n) {
            return edge;
          }
        }
        return nullptr;
      }
    };

    struct Edge {
      Node* node1{nullptr};
      Node* node2{nullptr};

      float length{0};
      float cost{0};
      int cls{0};

      Edge(Node* const n1, Node* const n2, float c = 0, int cls = 0) :
        node1(n1), node2(n2), cost(c), cls(cls)
      {
        length = (n1->pos - n2->pos).norm();
      }
      Edge() = default;

      float totalCost() {
        return length + 10*cost + std::pow(100, cls)-1;
      }

      Node* getOtherNode(const Node* n) const {
        if (n == node1) return node2;
        return node1;
      }
    };

    //! Djikstra shortest-path solver
    std::list<Node*> getPath(Node* const start_n, Node* const end_n);

    //! @return pointer to inserted node
    Node* addNode(const Node& node);

    void addEdge(const Edge& edge);

    auto& getNode(int ind) {
      return nodes_[ind];
    }

    const auto& getNodes() const {
      return nodes_;
    }

    const auto& getEdges() const {
      return edges_;
    }

    auto& getEdges() {
      return edges_;
    }

    int size() const {
      return nodes_.size();
    }

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void reset();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    std::map<int, Node> nodes_;
    std::list<Edge> edges_;

    Timer* get_path_t_;
};

} // namespace spomp
