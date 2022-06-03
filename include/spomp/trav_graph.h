#pragma once

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

      float cost{0};
      int cls{0};

      Edge(Node* const n1, Node* const n2, float c, int cls) :
        node1(n1), node2(n2), cost(c), cls(cls) {}

      Node* getOtherNode(const Node* n) const {
        if (n == node1) return node2;
        return node1;
      }
    };

    //! Djikstra shortest-path solver
    std::list<const Node*> getPath(Node* start_n, Node* end_n);

    void addNode(const Node& node) {
      nodes_.push_back(node);
    }

    void addEdge(const Edge& edge);

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void reset();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

} // namespace spomp
