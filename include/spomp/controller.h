#pragma once

#include <Eigen/Dense>
#include "spomp/utils.h"
#include "spomp/terrain_pano.h"

namespace spomp {

class Controller {
  public:
    struct Params {
    };
    Controller(const Params& params);

    Twistf getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
      const TerrainPano& pano);

  protected:
    Params params_;
};

} // namespace spomp
