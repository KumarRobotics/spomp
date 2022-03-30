#include "spomp/local.h"

namespace spomp {

Local::Local(const TerrainPano::Params& tp_p, const PanoPlanner::Params& pp_p) : 
  pano_(tp_p), planner_(pp_p) {}

void Local::updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose) {
  pano_.updatePano(pano, pose);
  planner_.updatePano(pano_);
}

} // namespace spomp
