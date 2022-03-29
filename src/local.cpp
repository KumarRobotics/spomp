#include "spomp/local.h"

namespace spomp {

Local::Local(const TerrainPano::Params& tp_p) : pano_(tp_p) {}

void Local::updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose) {
  pano_.updatePano(pano, pose);
}

} // namespace spomp
