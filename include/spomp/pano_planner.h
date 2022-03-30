#pragma once

#include "spomp/terrain_pano.h"

namespace spomp {

class PanoPlanner {
  public:
    struct Params {
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
    Eigen::VectorXi reachability_;
};

} // namespace spomp
