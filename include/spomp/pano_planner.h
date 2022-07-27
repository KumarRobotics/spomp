#pragma once

#include "spomp/terrain_pano.h"

namespace spomp {

class PanoPlanner {
  public:
    struct Params {
      int tbb = -1;
      float max_spacing_m = 0.5;
      int sample_size = 100;
      float consistency_cost = 0.2;
    };

    PanoPlanner(const Params& params);

    void updatePano(const TerrainPano& pano);

    /*!
     * Compute feasible local goal based off of global goal
     * @param goal The goal in the pano frame
     * @return The local goal in the pano frame
     */
    Eigen::Vector2f plan(const Eigen::Vector2f& goal, 
        const Eigen::Vector2f& old_goal = Eigen::Vector2f::Zero()) const;

    const auto& getReachability() const {
      return reachability_;
    }

    //! Get the range of reachability at the given azimuth (in radians)
    float getRangeAtAz(float az) const {
      return reachability_.scan[reachability_.proj.indAt(az)];
    }

    //! @return True if point is within reachable area
    bool isSafe(const Eigen::Vector2f& pt) const;

  protected:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    //! Vector of distances with respect to angle
    Reachability reachability_;

    // Timers
    Timer* pano_update_t_{};
};

} // namespace spomp
