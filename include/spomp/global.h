#pragma once

#include "spomp/trav_map.h"

namespace spomp {

class Global {
  public:
    Global(const TravMap::Params& tm_p);

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
     * @return The next global target waypoint, if available
     */
    std::optional<Eigen::Vector2f> getNextWaypoint(
        const Eigen::Isometry3f& state);

  private:
    TravMap map_;
};

} // namespace spomp
