#pragma once

#include <Eigen/Dense>
#include "spomp/terrain_pano.h"
#include "spomp/trav_graph.h"
#include "spomp/utils.h"

namespace spomp {

class Reachability {
  public:
    Reachability(const Eigen::VectorXf& scan, const Eigen::VectorXi& is_obs,
        const AngularProj& proj, const Eigen::Isometry2f& pose);
    Reachability(const AngularProj& proj, const Eigen::Isometry2f& pose);
    Reachability() = default;

    float getRangeAtAz(float az) const {
      return scan_[proj_.indAt(az)];
    }

    int size() const {
      return scan_.size();
    }

    void setAzTrav(int ind, float dist, bool is_obs);

    float maxRange() const;

    enum EdgeExperience {TRAV, NOT_TRAV, UNKNOWN};
    EdgeExperience analyzeEdge(const Eigen::Vector2f& start_p, 
        const Eigen::Vector2f& end_p) const;

    // Setters and getters
    const auto& getPose() const {
      return pose_;
    }
    void setScan(const Eigen::VectorXf& scan) {
      scan_ = scan;
    }
    void setObs(const Eigen::VectorXi& is_obs) {
      is_obs_ = is_obs;
    }

  private:
    Eigen::VectorXf scan_{};
    Eigen::VectorXi is_obs_{};
    AngularProj proj_{};
    Eigen::Isometry2f pose_{Eigen::Isometry2f::Identity()};
};

} // namespace spomp
