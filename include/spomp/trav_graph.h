#pragma once

#include <iostream>
#include <map>
#include <list>
#include <Eigen/Dense>
#include "spomp/utils.h"
#include "spomp/reachability.h"
#include "spomp/timer.h"

namespace spomp {

/*!
 * Traversability graph representation
 */
class TravGraph {
  public:
    struct Params {
      float reach_node_max_dist_m = 4;
      float trav_window_rad = 0.3;
      float max_trav_discontinuity_m = 2;
      int num_edge_exp_before_mark = 2;
      float trav_edge_prob_trav = 0.99;
    };
    TravGraph(const Params& p);

    // Forward declaration
    struct Edge;
    struct Node {
      int id{0};
      std::vector<Edge*> edges{};
      Eigen::Vector2f pos{0, 0};

      // Graph-search temps
      bool visited{false};
      double cost{std::numeric_limits<double>::infinity()};
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
      static int MAX_TERRAIN;

      Node* node1{nullptr};
      Node* node2{nullptr};

      float length{0};
      double cost{0};
      int cls{0};
      bool is_locked{false};
      bool is_experienced{false};
      int untrav_counter{0};

      Edge(Node* const n1, Node* const n2, float c = 0, int cls = 0) :
        node1(n1), node2(n2), cost(c), cls(cls)
      {
        length = (n1->pos - n2->pos).norm();
      }
      Edge() = default;

      double totalCost() {
        return length * cost + ((std::pow(100, cls)-1) * (length + 1));
      }

      Node* getOtherNode(const Node* n) const {
        if (n == node1) return node2;
        return node1;
      }

      void incUntravCounter() {
        is_experienced = true;
        if (!is_locked) {
          // Reset counter if not locked.  If locked, then want
          // it to be even harder to mark
          untrav_counter = std::max(0, untrav_counter);
        }
        untrav_counter += 1;
      }

      void decUntravCounter() {
        is_experienced = true;
        if (!is_locked) {
          untrav_counter = std::min(0, untrav_counter);
        }
        untrav_counter -= 1;
      }
    };

    //! Djikstra shortest-path solver
    std::list<Node*> getPath(Node* const start_n, Node* const end_n);

    double getPathCost(const std::list<Node*>& path) const;
    float getPathLength(const std::list<Node*>& path) const;

    //! @return True if map changed
    bool updateLocalReachability(const Reachability& reachability);
    bool updateEdgeFromReachability(TravGraph::Edge& edge, 
        const TravGraph::Node& start_node, const Reachability& reachability,
        std::optional<Eigen::Vector2f> start_pos = {});

    //! @return pointer to inserted node
    Node* addNode(const Node& node);

    //! @return pointer to inserted edge
    Edge* addEdge(const Edge& edge);

    auto& getNode(int ind) {
      return nodes_[ind];
    }

    const auto& getNodes() const {
      return nodes_;
    }

    auto& getNodes() {
      return nodes_;
    }

    const std::vector<Node*> getNodesNear(const Eigen::Vector2f& pos, float delta=1);

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

    void verifyCanExitNode(Node* const node);

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    Params params_;

    std::map<int, Node> nodes_;
    std::list<Edge> edges_;

    Timer* get_path_t_;
    Timer* update_edge_t_;
    Timer* get_near_nodes_t_;
};

} // namespace spomp
