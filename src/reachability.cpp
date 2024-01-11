#include "spomp/reachability.h"
#include <iostream>

namespace spomp {

/**

* @class Reachability
* @brief Represents the reachability of a pose in a 2D space based on a scan
*/
    Reachability::Reachability(uint64_t stamp, const Eigen::VectorXf& scan, const Eigen::VectorXi& is_obs,
                               const AngularProj& proj, const Eigen::Isometry2f& pose) :
  stamp_(stamp),
  scan_(scan),
  is_obs_(is_obs),
  proj_(proj),
  pose_(pose) {}

/**
 * @class Reachability
 * @brief Represents the reachability of a pose in a certain angular projection.
 */
    Reachability::Reachability(uint64_t stamp, const AngularProj& proj,
                               const Eigen::Isometry2f& pose) :
  stamp_(stamp),
  proj_(proj),
  pose_(pose)
{
  scan_ = Eigen::VectorXf::Zero(proj_.num);
  is_obs_ = Eigen::VectorXi::Zero(proj_.num);
}

/**
 * @brief Sets the azimuth traversal distance and obstacle flag at the given index.
 *
 * This function sets the azimuth traversal distance and obstacle flag at the specified index
 * in the scan array.
 *
 * @param ind The index at which to set the azimuth traversal distance and obstacle flag.
 * @param dist The azimuth traversal distance to set.
 * @param is_obs The obstacle flag to set.
 *
 * @note The index must be within the valid range of the scan array, i.e., between 0 and size - 1.
 */
    void Reachability::setAzTrav(int ind, float dist, bool is_obs) {
  if (ind >= 0 && ind < size()) {
    scan_[ind] = dist;
    is_obs_[ind] = is_obs;
  }
}

/**
 * @brief Calculates the maximum range value in the given scan.
 *
 * This function calculates the maximum range value in the current scan data.
 * If the size of the scan is less than 1, the function returns 0.
 *
 * @return The maximum range value in the scan.
 */
    float Reachability::maxRange() const {
  if (size() < 1) return 0;
  return scan_.maxCoeff();
}

/**
 * @brief Checks if a given point is inside the reachability region.
 *
 * This function takes a point in the Cartesian coordinate system and checks if it lies
 * inside the reachability region. It first converts the point to the local coordinate
 * system using the inverse of the pose, and then converts the point to polar coordinates.
 * It then retrieves the obstacle at the azimuth angle of the polar coordinate and checks
 * if the distance from the origin is less than or equal to the range of the obstacle. In
 * order to always consider the origin as inside the reachability region, the distance
 * comparison is performed using <= operator.
 *
 * @param pt The point to be checked, given in the Cartesian coordinate system.
 * @return True if the point is inside the reachability region, false otherwise.
 */
    bool Reachability::pointInside(const Eigen::Vector2f& pt) const {
  Eigen::Vector2f local_pt = pose_.inverse() * pt;
  Eigen::Vector2f local_pt_polar = cart2polar(local_pt);

  auto obs = getObsAtAz(local_pt_polar[1]);
  // Important to be <= since we want origin to always be considered inside
  return local_pt_polar[0] <= obs.range;
}

/**
 * @brief Determines the experience of an edge based on the given start and end points.
 *
 * This function analyzes an edge defined by the given start and end points to determine its experience.
 * The experience can be one of the following:
 *  - TRAV: The edge is traversable.
 *  - NOT_TRAV: The edge is not traversable.
 *  - UNKNOWN: The experience of the edge cannot be determined.
 *
 * @param start_p The start point of the edge.
 * @param end_p The end point of the edge.
 * @param params The parameters for edge analysis.
 * @return The experience of the edge.
 */
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

  float crossing_az = std::numeric_limits<float>::quiet_NaN();
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
    // is correct.  Also check if crossing is between endpts
    if (b.isApprox(A * t) && (t.array() >= 0).all() && (t.array() <= 1).all()) {
      // We have a crossing
      Eigen::Vector2f cross_pt_polar = cart2polar(pt1 + (pt2 - pt1)*t[0]);
      if (std::isnan(crossing_az)) {
        crossing_az = cross_pt_polar[1];
      } else {
        // If there is more than one crossing, this suggests that the path goes 
        // out->in->out or something along this line.
        // This case could just be a glancing edge around a corner or something, so
        // we adopt a "wait-and-see" approach
        return UNKNOWN;
      }
    }
  }

  if (std::isnan(crossing_az)) {
    // If no crossings, either entirely inside or outside
    if (start_in && end_in) {
      // No crossings, not outside, so must be entirely inside
      return TRAV;
    } else {
      return UNKNOWN;
    }
  }

  // Determine which endpoint of edge is outside of the trav area
  Eigen::Vector2f outside_pt = local_end_p;
  if (end_in) {
    outside_pt = local_start_p;
  }

  float last_range = std::numeric_limits<float>::quiet_NaN();
  for (float az=crossing_az-params.trav_window_rad; az<=crossing_az+params.trav_window_rad; 
      az+=std::abs(proj_.delta_angle)) 
  {
    auto obs = getObsAtAz(az);
    if (std::isnan(last_range)) {
      last_range = obs.range;
    }
    if (!obs.is_obs || obs.range > outside_pt.norm()) {
      return UNKNOWN;
    }
    if (!std::isnan(last_range) && obs.range - last_range > params.max_trav_discontinuity_m) {
      // Major discontinuity in range suggests a possible gap
      return UNKNOWN;
    }
    last_range = obs.range;
  }

  return NOT_TRAV;
}

} // namespace spomp
