#include <algorithm>
#include <tbb/parallel_for.h>
#include "spomp/terrain_pano.h"

namespace spomp {

TerrainPano::TerrainPano(const Params& params) : params_(params) {
}

void TerrainPano::updatePano(const Eigen::ArrayXXf& pano, 
    const Eigen::Isometry3f& pose) 
{
  pano_ = pano;
  fillHoles(pano_);
  computeCloud();
  Eigen::ArrayXXf grad = computeGradient();
  traversability_pano_ = threshold(grad);
  inflate(traversability_pano_);
}

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
            if (col_i - last_nonzero > 1 && col_i - last_nonzero < params_.max_hole_fill_size && 
                last_nonzero >= 0) 
            {
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

void TerrainPano::computeCloud() {
  Eigen::VectorXf alts = Eigen::VectorXf::LinSpaced(pano_.rows(), 
      params_.v_fov_rad/2, -params_.v_fov_rad/2);
  Eigen::VectorXf azs = Eigen::VectorXf::LinSpaced(pano_.cols(), 
      0, 2*pi*(1 - 1./pano_.cols()));

  Eigen::VectorXf alts_c = alts.array().cos();
  Eigen::VectorXf alts_s = alts.array().sin();

  // Initialize
  for (auto& axis : cloud_) {
    axis = Eigen::ArrayXXf(pano_.rows(), pano_.cols());
  }

  int gsize = params_.tbb <= 0 ? pano_.cols() : params_.tbb;

  tbb::parallel_for(tbb::blocked_range<int>(0, pano_.cols(), gsize), 
    [&](tbb::blocked_range<int> range) {
      for (int col_i=range.begin(); col_i<range.end(); ++col_i) {
        // Loop through cols because Eigen stores col-major
        // This means we access more contiguous blocks of memory
        cloud_[0].col(col_i) = pano_.col(col_i) * alts_c.array();
        cloud_[1].col(col_i) = cloud_[0].col(col_i) * sin(azs(col_i));
        cloud_[0].col(col_i) = cloud_[0].col(col_i) * cos(azs(col_i));
        cloud_[2].col(col_i) = pano_.col(col_i) * alts_s.array();
      }
    });
}

//! Compute the gradient across the panorama
Eigen::ArrayXXf TerrainPano::computeGradient() const {
  Eigen::VectorXf alts = Eigen::VectorXf::LinSpaced(pano_.rows(), 
      params_.v_fov_rad/2, -params_.v_fov_rad/2);
  Eigen::VectorXf alts_c = alts.array().cos();
  Eigen::VectorXf alts_s = alts.array().sin();

  // LiDAR at ~0.4m off ground
  Eigen::VectorXf pred_ranges = 0.4 / (-alts).array().tan();

  // Compute horizonal gradient
  Eigen::ArrayXXf grad_h = Eigen::ArrayXXf::Zero(pano_.rows(), pano_.cols());
  Eigen::ArrayXXf grad_v = Eigen::ArrayXXf::Zero(pano_.rows(), pano_.cols());

  int gsize = params_.tbb <= 0 ? pano_.rows() : params_.tbb;
  tbb::parallel_for(tbb::blocked_range<int>(0, pano_.rows(), gsize), 
    [&](tbb::blocked_range<int> range) {
      for (int row_i=range.begin(); row_i<range.end(); ++row_i) {
        // Calculate vertical spacing
        int delta = 0;
        do {
          ++delta;
        } while (abs(pred_ranges[row_i] - pred_ranges[row_i - delta]) < params_.target_dist_xy &&
                 pred_ranges[row_i - delta] > 0 && row_i - (delta + 1) > 0);

        for (int col_i=0; col_i<pano_.cols(); ++col_i) {
          if (cloud_[2](row_i, col_i) != 0) {
            // Horizontal grad
            float xy_range = alts_c[row_i] * pano_(row_i, col_i);
            float arc_length = xy_range * 2 * pi / pano_.cols();
            int window = std::min<int>(params_.target_dist_xy / arc_length, pano_.cols() / 20);
            window = window / 2 + 1; // half size
            
            // Add cols to force positive
            int col_i1 = fast_mod(col_i - window + pano_.cols(), pano_.cols());
            int col_i2 = fast_mod(col_i + window, pano_.cols());

            float v_noise = params_.noise_m*alts_s[row_i];
            float h_noise = params_.noise_m*alts_c[row_i];
            if (cloud_[2](row_i, col_i1) != 0 && cloud_[2](row_i, col_i2) != 0) {
              grad_h(row_i, col_i) = std::max<float>(
                  abs(cloud_[2](row_i, col_i1) - cloud_[2](row_i, col_i2)) - v_noise, 0) / 
                  (arc_length * (window * 2 + 1));
            }
            
            // Vertical grad
            if (cloud_[2](row_i - delta, col_i) != 0) {
              grad_v(row_i, col_i) = std::max<float>(
                  abs(cloud_[2](row_i, col_i) - cloud_[2](row_i - delta, col_i)) - v_noise, 0) /
                std::max<float>(
                  abs(xy_range - alts_c[row_i - delta] * pano_(row_i - delta, col_i)) + h_noise, 0);
            }
          }
        }  
      }
    });

  // Combine gradients
  return (grad_h.pow(2) + grad_v.pow(2)).sqrt();
}

//! Threshold the gradient into obstacles and filter
Eigen::ArrayXXi TerrainPano::threshold(const Eigen::ArrayXXf& grad_pano) const {
  Eigen::ArrayXXi thresh_pano = (grad_pano > params_.slope_thresh).cast<int>();

  // Find small obstacles
  int gsize = params_.tbb <= 0 ? thresh_pano.rows() : params_.tbb;
  tbb::parallel_for(tbb::blocked_range<int>(0, thresh_pano.rows(), gsize), 
    [&](tbb::blocked_range<int> range) {
      for (int row_i=range.begin(); row_i<range.end(); ++row_i) {
        // This is very similar to the hole removal code, essentially the same idea
        // but in reverse
        int first_zero = -1;
        int last_zero = -1;
        // Loop around back to starting first nonzero to make sure we wrap around
        for (int col_i=0; col_i<=first_zero + thresh_pano.cols(); ++col_i) {
          int val = thresh_pano(row_i, fast_mod(col_i, thresh_pano.cols()));
          if (val == 0) {
            if (col_i - last_zero > 1 && col_i - last_zero <= params_.min_noise_size && 
                last_zero >= 0) 
            {
              // Found a hole small enough to fill
              for (int clear_col_i=last_zero+1; clear_col_i<col_i; ++clear_col_i) {
                thresh_pano(row_i, fast_mod(clear_col_i, thresh_pano.cols())) = 0;
              }
            }
            
            last_zero = col_i;
            if (first_zero < 0) {
              first_zero = last_zero;
            }
          }
        }
      }
    });

  return thresh_pano;
}

//! Inflate obstacles, modifies in place
void TerrainPano::inflate(Eigen::ArrayXXi& trav_pano) const {
}

} // namespace spomp
