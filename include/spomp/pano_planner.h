#pragma once

#include "spomp/terrain_pano.h"

namespace spomp {

class PanoPlanner {
  public:
    struct Params {
      int tbb = -1;
      float max_spacing_m = 0.5;
    };

    PanoPlanner(const Params& params);

    void updatePano(const TerrainPano& pano);

    /*!
     * Compute feasible local goal based off of global goal
     * @param goal The goal in the pano frame
     * @return The local goal in the pano frame
     */
    Eigen::Vector2f plan(const Eigen::Vector2f& goal) const;

    struct Reachability {
      Eigen::VectorXf scan{};
      AngularProj az_p;
    };
    const auto& getReachability() const {
      return reachability_;
    }

    //! Get the range of reachability at the given azimuth (in radians)
    float getRangeAtAz(float az) {
      return 0;
    }

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
