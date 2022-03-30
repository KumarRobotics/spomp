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

    const auto& getReachability() const {
      return reachability_;
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
    Eigen::VectorXf reachability_;

    // Timers
    Timer* pano_update_t_{};
};

} // namespace spomp
