#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace spomp {

/*!
 * Analyzes depth panoramas and computes traversability metrics
 */
class TerrainPano {
  public:
    struct Params {
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

    //! Fills small holes in the panorama, modifies pano in-place
    void fillHoles(Eigen::ArrayXXf& pano) const;

    //! Compute the gradient across the panorama
    Eigen::ArrayXXi computeGradient(const Eigen::ArrayXXf& pano) const;

    //! Threshold the gradient into obstacles and filter
    Eigen::ArrayXXi threshold(const Eigen::ArrayXXf& grad_pano) const;

    //! Inflate obstacles, modifies in place
    void inflate(Eigen::ArrayXXi& trav_pano) const;
    
  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void computeCloud();

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    const Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    Eigen::ArrayXXf pano_;
    Eigen::ArrayXXi traversability_pano_;

    std::array<Eigen::ArrayXXf, 3> cloud_;
};

} // namespace spomp
