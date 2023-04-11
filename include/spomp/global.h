#pragma once

#include "spomp/trav_map.h"
#include "spomp/waypoint_manager.h"
#include "spomp/utils.h"

namespace spomp {

class Global {
  public:
    struct Params {
      int max_num_recovery_reset = 1;
      float timeout_duration_s_per_m = 3;
      float replan_hysteresis = 10;
      int num_new_reach_to_pub = 10;
    };
    Global(const Params& g_p, 
           const TravMap::Params& tm_p, 
           const TravGraph::Params& tg_p, 
           const AerialMapInfer::Params& am_p, 
           const MLPModel::Params& mlp_p, 
           const WaypointManager::Params& wm_p);

    void updateMap(const cv::Mat &map, const Eigen::Vector2f& center,
        const std::vector<cv::Mat>& other_maps = {}) 
    {
      map_.updateMap(map, center, other_maps);
    }

    /*!
     * Set the current global goal
     * @return true if the goal can be reached
     */
    bool setGoal(const Eigen::Vector3f& goal);

    /*! 
     * @param state The current global pose of the robot in map frame
     * @return Current manager state
     */
    auto setState(const Eigen::Isometry3f& state) {
      return waypoint_manager_.setState(state.translation().head<2>());
    }

    float getTimeoutDuration() const {
      float total_length = map_.getPathLength(waypoint_manager_.getPath());
      if (total_length < std::numeric_limits<float>::max()) {
        return params_.timeout_duration_s_per_m * total_length;
      }
      // This is a bad scenario, try again soon
      return params_.timeout_duration_s_per_m * 5;
    }

    void cancel() {
      waypoint_manager_.cancel();
    }

    /*!
     * @return false if plan failed
     */
    bool updateLocalReachability(const Reachability& reachability);
    bool updateOtherLocalReachability(
        const Reachability& reachability, int robot_id);

    void resetGraphLocked() {
      map_.resetGraphLocked();
    }

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

    const auto& getReachabilityHistory() {
      const auto& reach_hist = map_.getReachabilityHistory();
      num_reach_panos_published_ = reach_hist.size();
      return reach_hist;
    }

    bool haveNewReachabilityHistory() const {
      return map_.getReachabilityHistory().size() >=
        num_reach_panos_published_ + params_.num_new_reach_to_pub;
    }

    auto getPos() const {
      return waypoint_manager_.getPos();
    }

    bool haveReachabilityForRobotAtStamp(int robot_id, uint64_t stamp) const {
      return map_.haveReachabilityForRobotAtStamp(robot_id, stamp);
    }

  private:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    TravMap map_;
    WaypointManager waypoint_manager_;

    float old_cost_{0};
    int num_reach_panos_published_{0};
};

} // namespace spomp
