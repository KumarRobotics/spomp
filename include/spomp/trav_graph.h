#pragma once

#include <iostream>
#include <list>
#include <Eigen/Dense>

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

      float totalCost() {
        return length + cost + std::pow(100, cls)-1;
      }

      Node* getOtherNode(const Node* n) const {
        if (n == node1) return node2;
        return node1;
      }
    };

    //! Djikstra shortest-path solver
    std::list<const Node*> getPath(Node* const start_n, Node* const end_n);

    //! @return index of inserted node
    Node* addNode(const Node& node);

    void addEdge(const Edge& edge);

    auto& getNodes() {
      return nodes_;
    }

    auto& getEdges() {
      return edges_;
    }

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void reset();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    std::list<Node> nodes_;
    std::list<Edge> edges_;
};

} // namespace spomp
