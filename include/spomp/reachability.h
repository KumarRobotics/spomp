#pragma once

#include <Eigen/Dense>
#include "spomp/terrain_pano.h"
#include "spomp/utils.h"

namespace spomp {

class Reachability {
  public:
    Reachability(const Eigen::VectorXf& scan, const Eigen::VectorXi& is_obs,
        const AngularProj& proj, const Eigen::Isometry2f& pose);
    Reachability(const AngularProj& proj, const Eigen::Isometry2f& pose);
    Reachability() = default;

    struct RangeObs {
      float range;
      bool is_obs;
    };
    RangeObs getObsAtAz(float az) const {
      int ind = proj_.indAt(az);
      return {scan_[ind], static_cast<bool>(is_obs_[ind])};
    }

    int size() const {
      return scan_.size();
    }

    void setAzTrav(int ind, float dist, bool is_obs);

    float maxRange() const;

    enum EdgeExperience {TRAV, NOT_TRAV, UNKNOWN};
    struct EdgeAnalysisParams {
      float trav_window_rad;
      float reach_max_dist_to_be_obs_m;
    };
    EdgeExperience analyzeEdge(const Eigen::Vector2f& start_p, 
        const Eigen::Vector2f& end_p, const EdgeAnalysisParams& params) const;

    // Setters and getters
    const auto& getScan() const {
      return scan_;
    }
    const auto& getIsObs() const {
      return is_obs_;
    }
    const auto& getProj() const {
      return proj_;
    }
    const auto& getPose() const {
      return pose_;
    }
    void setScan(const Eigen::VectorXf& scan) {
      scan_ = scan;
    }
    void setIsObs(const Eigen::VectorXi& is_obs) {
      is_obs_ = is_obs;
    }

  private:
    Eigen::VectorXf scan_{};
    Eigen::VectorXi is_obs_{};
    AngularProj proj_{};
    Eigen::Isometry2f pose_{Eigen::Isometry2f::Identity()};
};

} // namespace spomp
