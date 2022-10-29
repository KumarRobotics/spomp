#include "spomp/reachability.h"
#include <iostream>

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

bool Reachability::pointInside(const Eigen::Vector2f& pt) const {
  Eigen::Vector2f local_pt = pose_.inverse() * pt;
  Eigen::Vector2f local_pt_polar = cart2polar(local_pt);

  auto obs = getObsAtAz(local_pt_polar[1]);
  return local_pt_polar[0] < obs.range;
}

Reachability::EdgeExperience Reachability::analyzeEdge(const Eigen::Vector2f& start_p, 
    const Eigen::Vector2f& end_p, const EdgeAnalysisParams& params) const 
{
  bool start_in = pointInside(start_p);
  bool end_in = pointInside(end_p);
  if (!start_in && !end_in) {
    // Both endpoints are outside trav region
    return UNKNOWN;
  }

  Eigen::Vector2f local_start_p = pose_.inverse() * start_p;
  Eigen::Vector2f local_end_p = pose_.inverse() * end_p;

  std::vector<float> crossing_az;

  Eigen::Matrix2f A;
  A.col(1) = local_end_p - local_start_p;
  Eigen::Vector2f b;
  for (int edge_ind=0; edge_ind<size(); ++edge_ind) {
    int edge_ind2 = fast_mod(edge_ind+1, size());
    Eigen::Vector2f pt1 = polar2cart({scan_[edge_ind], proj_.angAt(edge_ind)});
    Eigen::Vector2f pt2 = polar2cart({scan_[edge_ind2], proj_.angAt(edge_ind2)});
    A.col(0) = pt1 - pt2;
    b = pt1 - local_start_p;

    Eigen::Vector2f t = A.householderQr().solve(b);
    // Householder QR will always return something, but have to check whether solution
    // is correct as well as whether intersection point is on line segs
    if ((t.array() >= 0).all() && (t.array() <= 1).all() && b.isApprox(A * t)) {
      // We have a crossing
      Eigen::Vector2f cross_pt_polar = cart2polar(pt1 + (pt2 - pt1)*t[0]);
      crossing_az.push_back(cross_pt_polar[1]);
    }
  }

  if (crossing_az.size() == 0) {
    if (start_in && end_in) {
      // No crossings, not outside, so must be entirely inside
      return TRAV;
    } else {
      return UNKNOWN;
    }
  }

  for (float c_az : crossing_az) {
    for (float az=c_az-params.trav_window_rad; az<=c_az+params.trav_window_rad; 
        az+=std::abs(proj_.delta_angle)) 
    {
      if (!getObsAtAz(az).is_obs) {
        return UNKNOWN;
      }
    }
  }

  return NOT_TRAV;
}

} // namespace spomp
