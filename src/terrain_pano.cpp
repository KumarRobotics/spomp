#include "spomp/terrain_pano.h"

namespace spomp {

TerrainPano::TerrainPano(const Params& params) : params_(params) {}

void TerrainPano::updatePano(const Eigen::ArrayXXf& pano, 
    const Eigen::Isometry3f& pose) 
{}

void TerrainPano::fillHoles(Eigen::ArrayXXf& pano) const {
  for (int row_i=0; row_i<pano.rows(); ++row_i) {
    int first_nonzero = -1;
    int last_nonzero = -1;
    for (int col_i=0; col_i<pano.cols(); ++col_i) {
      if (pano(row_i, col_i) > 0) {
        // Found a hole small enough to fill
        if (last_nonzero - row_i > 1 && last_nonzero - row_i < 100) {
        }
        
        last_nonzero = col_i;
        if (first_nonzero < 0) {
          first_nonzero = last_nonzero;
        }
      }
    }
  }
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
