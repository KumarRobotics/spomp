#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "spomp/utils.h"
#include "spomp/timer.h"

namespace spomp {

/*!
 * Analyzes depth panoramas and computes traversability metrics
 */
class TerrainPano {
  public:
    struct Params {
      int tbb = -1;
      float max_hole_fill_size = 0.1;
      float min_noise_size = 0.3;
      float v_fov_rad = deg2rad(90);
      float target_dist_xy = 0.5;
      float noise_m = 0.05;
      float slope_thresh = 0.3;
      float inflation_m = 0.5;
    };

    TerrainPano(const Params& params);

    //! Update the internal depth panorama
    void updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose);

    const Eigen::ArrayXXi& getTraversability() const {
      return traversability_pano_;
    }

    const Eigen::ArrayXXf& getPano() const {
      return pano_;
    }

    const std::array<Eigen::ArrayXXf, 3>& getCloud() const {
      return cloud_;
    }
    
  protected:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    //! Fills small holes in the panorama, modifies pano in-place
    void fillHoles(Eigen::ArrayXXf& pano) const;

    //! Compute cartesian point for each point in the cloud
    void computeCloud();

    //! Compute the gradient across the panorama
    Eigen::ArrayXXf computeGradient() const;

    //! Threshold the gradient into obstacles and filter
    Eigen::ArrayXXi threshold(const Eigen::ArrayXXf& grad_pano) const;

    //! Inflate obstacles, modifies in place
    void inflate(Eigen::ArrayXXi& trav_pano) const;

    //! Get the window size for a given distance
    int getWindow(float dist_m, int row_i, int col_i) const;

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    Eigen::ArrayXXf pano_;
    Eigen::ArrayXXi traversability_pano_;

    std::array<Eigen::ArrayXXf, 3> cloud_;

    //! Timers
    Timer* pano_update_t_{};
    Timer* fill_holes_t_{};
    Timer* compute_cloud_t_{};
    Timer* compute_grad_t_{};
    Timer* thresh_t_{};
    Timer* inflate_t_{};
};

} // namespace spomp
