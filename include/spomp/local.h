#pragma once

#include "spomp/terrain_pano.h"

namespace spomp {

class Local {
  public:
    Local(const TerrainPano::Params& tp_p);

    void updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose);

    const TerrainPano& getPano() const {
      return pano_;
    }

  protected:
    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    TerrainPano pano_;
};

} // namespace spomp
