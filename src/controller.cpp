#include "spomp/controller.h"

namespace spomp {

Controller::Controller(const Params& params) : params_(params) {}

Twistf Controller::getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
  const TerrainPano& pano)
{
  return {};
}

} // namespace spomp
