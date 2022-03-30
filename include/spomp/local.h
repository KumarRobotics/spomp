#pragma once

#include "spomp/terrain_pano.h"
#include "spomp/pano_planner.h"

namespace spomp {

class Local {
  public:
    Local(const TerrainPano::Params& tp_p, const PanoPlanner::Params& pp_p);

    void updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose);

    const TerrainPano& getPano() const {
      return pano_;
    }

    const PanoPlanner& getPlanner() const {
      return planner_;
    }

  protected:
    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    TerrainPano pano_;
    PanoPlanner planner_;
};

} // namespace spomp
