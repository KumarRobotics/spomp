#pragma once

#include <Eigen/Dense>

namespace spomp {

/*!
 * Traversability graph representation
 */
class TravGraph {
  public:
    TravGraph();

    std::vector<const Node*> getPath();

  private:
    struct Node {
      std::vector<Edge*> edges;
      
      Eigen::Vector2f pos{0, 0};
    };

    struct Edge {
      Node* node1;
      Node* node2;

      float cost{0};
      int cls{0};
    };

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

} // namespace spomp
