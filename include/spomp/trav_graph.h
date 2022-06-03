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

      Edge(Node* const n1, Node* const n2, float c, int cls) :
        node1(n1), node2(n2), cost(c), cls(cls) 
      {
        length = (n1->pos - n2->pos).norm();
      }

      float totalCost() {
        return length + cost + 1e6 * cls;
      }

      Node* getOtherNode(const Node* n) const {
        if (n == node1) return node2;
        return node1;
      }
    };

    //! Djikstra shortest-path solver
    std::list<const Node*> getPath(int start_id, int end_id);

    //! @return index of inserted node
    int addNode(const Node& node);

    void addEdge(int n1_id, int n2_id, float c = 0, int cls = 0);

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
