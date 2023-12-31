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
      float max_distance_m = 1.0;
    };

    TerrainPano(const Params& params);

    //! Update the internal depth panorama
    void updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose);

    int rows() const {
      return pano_.rows();
    }

    int cols() const {
      return pano_.cols();
    }

    float rangeAt(int row, int col) const {
      return pano_(row, col) * alts_c_[row];
    }

    bool traversableAt(int row, int col) const {
      return traversability_pano_(row, col) >= params_.inflation_m;
    }

    float getObstacleDistAt(const Eigen::Vector2f& pt) const;

    const auto& getTraversability() const {
      return traversability_pano_;
    }

    const auto& getPano() const {
      return pano_;
    }

    const auto& getCloud() const {
      return cloud_;
    }

    const auto& getAzs() const {
      return azs_;
    }

    const auto& getAlts() const {
      return alts_;
    }

    const auto& getAzProj() const {
      return az_p_;
    }

    const auto& getAltProj() const {
      return alt_p_;
    }

    const auto& getPose() const {
      return pose_;
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

    /*! 
     * Generate distance transform
     * Return 0 at obstacle, then distance away up to max_dist
     */
    Eigen::ArrayXXf distance(const Eigen::ArrayXXi& obs_pano) const;

    //! Get the window size for a given distance
    static int getWindow(float dist_m, int row_i, int col_i, 
        const Eigen::ArrayXXf& pano);

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    AngularProj alt_p_, az_p_;
    Eigen::VectorXf alts_, azs_, alts_c_, alts_s_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    Eigen::ArrayXXf pano_;
    Eigen::ArrayXXf traversability_pano_;

    std::array<Eigen::ArrayXXf, 3> cloud_;

    Eigen::Isometry3f pose_{Eigen::Isometry3f::Identity()};

    // Timers
    Timer* pano_update_t_{};
    Timer* fill_holes_t_{};
    Timer* compute_cloud_t_{};
    Timer* compute_grad_t_{};
    Timer* thresh_t_{};
    Timer* dist_t_{};
};

} // namespace spomp
