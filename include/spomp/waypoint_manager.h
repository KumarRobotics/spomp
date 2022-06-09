#pragma once

#include "spomp/trav_graph.h"

namespace spomp {

class WaypointManager {
  public:
    struct Params {
      float waypoint_thresh_m = 2;
    };
    WaypointManager(const Params& p);

    void setState(const Eigen::Vector2f& pos);
    void setPath(const std::list<TravGraph::Node*>& path);

    std::optional<Eigen::Vector2f> getPos() {
      return robot_pos_;
    }

    std::optional<Eigen::Vector2f> getNextWaypoint();

    const auto& getPath() {
      return path_;
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
