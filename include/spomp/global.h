#pragma once

#include "spomp/trav_map.h"
#include "spomp/waypoint_manager.h"
#include "spomp/utils.h"

namespace spomp {

class Global {
  public:
    Global(const TravMap::Params& tm_p, const TravGraph::Params& tg_p, 
        const AerialMap::Params& am_p, const MLPModel::Params& mlp_p,
        const WaypointManager::Params& wm_p);

    void updateMap(const cv::Mat &map, const Eigen::Vector2f& center) {
      map_.updateMap(map, center);
    }

    /*!
     * Set the current global goal
     * @return true if the goal can be reached
     */
    bool setGoal(const Eigen::Vector3f& goal);

    /*! 
     * @param state The current global pose of the robot in map frame
     * @return true if path was just completed
     */
    bool setState(const Eigen::Isometry3f& state) {
      return waypoint_manager_.setState(state.translation().head<2>());
    }

    void cancel() {
      waypoint_manager_.cancel();
    }

    void updateLocalReachability(const Reachability& reachability);
    void updateOtherLocalReachability(const Reachability& reachability);

    //! @return The next global target waypoint, if available
    std::optional<Eigen::Vector2f> getNextWaypoint() {
      auto next_waypt = waypoint_manager_.getNextWaypoint();
      if (next_waypt) {
        return next_waypt->pos;
      }
      return {};
    }

    const auto& getEdges() const {
      return map_.getEdges();
    }

    const auto& getPath() const {
      return waypoint_manager_.getPath();
    }

    const cv::Mat getMapImageViz() const {
      return map_.viz();
    }

    const cv::Mat getAerialMapTrav() {
      return map_.getAerialMapTrav();
    }

    const auto& getReachabilityHistory() const {
      return reachability_history_;
    }

  private:
    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    TravMap map_;
    WaypointManager waypoint_manager_;

    std::vector<Reachability> reachability_history_;
};

} // namespace spomp
