#pragma once

#include <Eigen/Dense>
#include "spomp/terrain_pano.h"
#include "spomp/utils.h"

namespace spomp {

class Reachability {
  public:
    Reachability(uint64_t stamp, const Eigen::VectorXf& scan, 
        const Eigen::VectorXi& is_obs, const AngularProj& proj,
        const Eigen::Isometry2f& pose = Eigen::Isometry2f::Identity());
    Reachability(uint64_t stamp, const AngularProj& proj,
        const Eigen::Isometry2f& pose = Eigen::Isometry2f::Identity());
    Reachability() = default;

    struct RangeObs {
      float range;
      bool is_obs;
    };
    RangeObs getObsAtAz(float az) const {
      return getObsAtInd(proj_.indAt(az));
    }
    RangeObs getObsAtInd(int ind) const {
      return {scan_[ind], static_cast<bool>(is_obs_[ind])};
    }

    int size() const {
      return scan_.size();
    }

    void setAzTrav(int ind, float dist, bool is_obs);

    float maxRange() const;

    enum EdgeExperience {TRAV, NOT_TRAV, UNKNOWN};
    struct EdgeAnalysisParams {
      float trav_window_rad{0.2};
      float max_trav_discontinuity_m{0};
    };
    EdgeExperience analyzeEdge(const Eigen::Vector2f& start_p, 
        const Eigen::Vector2f& end_p, const EdgeAnalysisParams& params) const;

    bool pointInside(const Eigen::Vector2f& pt) const;

    // Setters and getters
    const auto& getScan() const {
      return scan_;
    }
    const auto& getIsObs() const {
      return is_obs_;
    }
    auto& getScan() {
      return scan_;
    }
    auto& getIsObs() {
      return is_obs_;
    }
    const auto& getProj() const {
      return proj_;
    }
    const auto& getPose() const {
      return pose_;
    }
    uint64_t getStamp() const {
      return stamp_;
    }
    bool isOtherRobot() const {
      return is_other_robot_;
    }
    void setPose(const Eigen::Isometry2f& pose) {
      pose_ = pose;
    }
    void setScan(const Eigen::VectorXf& scan) {
      scan_ = scan;
    }
    void setIsObs(const Eigen::VectorXi& is_obs) {
      is_obs_ = is_obs;
    }
    void setIsOtherRobot() {
      is_other_robot_ = true;
    }

  private:
    uint64_t stamp_{};
    bool is_other_robot_{false};
    Eigen::VectorXf scan_{};
    Eigen::VectorXi is_obs_{};
    AngularProj proj_{};
    Eigen::Isometry2f pose_{Eigen::Isometry2f::Identity()};
};

} // namespace spomp
