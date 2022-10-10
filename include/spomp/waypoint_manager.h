#pragma once

#include "spomp/trav_graph.h"

namespace spomp {

class WaypointManager {
  public:
    struct Params {
      float waypoint_thresh_m = 2;
    };
    WaypointManager(const Params& p);

    bool setState(const Eigen::Vector2f& pos);
    void setPath(const std::list<TravGraph::Node*>& path);

    std::optional<Eigen::Vector2f> getPos() const {
      return robot_pos_;
    }

    //! @return True if reached the end
    bool advancePlan();

    TravGraph::Node* getNextWaypoint() const;
    TravGraph::Node* getLastWaypoint() const;

    const auto& getPath() const {
      return path_;
    }

    TravGraph::Edge* getCurEdge() {
      return cur_edge_;
    }

    bool havePath() const {
      return path_.size() > 0;
    }

    void cancel() {
      path_.clear();
    }

  private:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    std::optional<Eigen::Vector2f> robot_pos_{};

    std::list<TravGraph::Node*> path_;
    std::list<TravGraph::Node*>::iterator next_node_{};
    TravGraph::Edge* cur_edge_{nullptr};
};

} // namespace spomp
