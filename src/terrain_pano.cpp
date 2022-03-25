#include "spomp/terrain_pano.h"

namespace spomp {

TerrainPano::TerrainPano(const Params& params) : params_(params) {}

void TerrainPano::updatePano(const Eigen::ArrayXXf& pano, 
    const Eigen::Isometry3f& pose) 
{
}

void TerrainPano::fillHoles(Eigen::ArrayXXf& pano) const {
}

//! Compute the gradient across the panorama
Eigen::ArrayXXi TerrainPano::computeGradient(const Eigen::ArrayXXf& pano) const {
  return {};
}

//! Threshold the gradient into obstacles and filter
Eigen::ArrayXXi TerrainPano::threshold(const Eigen::ArrayXXf& grad_pano) const {
  return {};
}

//! Inflate obstacles, modifies in place
void TerrainPano::inflate(Eigen::ArrayXXi& trav_pano) const {
}

} // namespace spomp
