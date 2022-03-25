#include <tbb/parallel_for.h>
#include "spomp/terrain_pano.h"
#include "spomp/utils.h"

namespace spomp {

TerrainPano::TerrainPano(const Params& params) : params_(params) {}

void TerrainPano::updatePano(const Eigen::ArrayXXf& pano, 
    const Eigen::Isometry3f& pose) 
{}

void TerrainPano::fillHoles(Eigen::ArrayXXf& pano) const {
  int gsize = params_.tbb <= 0 ? pano.rows() : params_.tbb;

  tbb::parallel_for(tbb::blocked_range<int>(0, pano.rows(), gsize), 
    [&](tbb::blocked_range<int> range) {
      for (int row_i=range.begin(); row_i<range.end(); ++row_i) {
        int first_nonzero = -1;
        int last_nonzero = -1;
        // Loop around back to starting first nonzero to make sure we wrap around holes
        for (int col_i=0; col_i<=first_nonzero + pano.cols(); ++col_i) {
          float val = pano(row_i, fast_mod(col_i, pano.cols()));
          if (val > 0) {
            if (col_i - last_nonzero > 1 && col_i - last_nonzero < 100 && last_nonzero >= 0) {
              // Found a hole small enough to fill
              float last_val = pano(row_i, fast_mod(last_nonzero, pano.cols()));
              for (int fill_col_i=last_nonzero+1; fill_col_i<col_i; ++fill_col_i) {
                // Linear interp
                pano(row_i, fast_mod(fill_col_i, pano.cols())) = 
                  ((fill_col_i - last_nonzero) * val + (col_i - fill_col_i) * last_val) /
                  (col_i - last_nonzero);
              }
            }
            
            last_nonzero = col_i;
            if (first_nonzero < 0) {
              first_nonzero = last_nonzero;
            }
          }
        }
      }
    });
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
