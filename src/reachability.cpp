#include "spomp/reachability.h"

namespace spomp {

Reachability::Reachability(const Eigen::VectorXf& scan, const Eigen::VectorXi& is_obs, 
    const AngularProj& proj, const Eigen::Isometry2f& pose) :
  scan_(scan),
  is_obs_(is_obs),
  proj_(proj),
  pose_(pose) {}

Reachability::Reachability(const AngularProj& proj, const Eigen::Isometry2f& pose) :
  proj_(proj),
  pose_(pose)
{
  scan_ = Eigen::VectorXf::Zero(proj_.num);
  is_obs_ = Eigen::VectorXi::Zero(proj_.num);
}

void Reachability::setAzTrav(int ind, float dist, bool is_obs) {
  if (ind >= 0 && ind < size()) {
    scan_[ind] = dist;
    is_obs_[ind] = is_obs;
  }
}

float Reachability::maxRange() const {
  if (size() < 1) return 0;
  return scan_.maxCoeff();
}

Reachability::EdgeExperience Reachability::analyzeEdge(const Eigen::Vector2f& start_p, 
    const Eigen::Vector2f& end_p, const EdgeAnalysisParams& params) const 
{
  Eigen::Vector2f local_dest_pose = pose_.inverse() * end_p;

  float range = local_dest_pose.norm();
  float bearing = atan2(local_dest_pose[1], local_dest_pose[0]);

  bool not_reachable = true;
  bool reachable = true;
  for (float b=bearing-params.trav_window_rad; b<=bearing+params.trav_window_rad; 
       b+=std::abs(proj_.delta_angle)) 
  {
    auto obs = getObsAtAz(b);
    if (range <= obs.range || !obs.is_obs) {
      // We have a non-obstacle path
      not_reachable = false;
    }
    if (range > obs.range) {
      // We have an obstacle or unknown path
      reachable = false;
      if (obs.range > params.reach_max_dist_to_be_obs_m) {
        // We think we can't get there, but it is far away.  Unclear.
        not_reachable = false;
      }
    }
  }

  if (reachable) {
    return TRAV;
  } else if (not_reachable) {
    return NOT_TRAV;
  }
  return UNKNOWN;
}

} // namespace spomp
