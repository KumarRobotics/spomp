#include <random>
#include <tbb/parallel_for.h>
#include "spomp/pano_planner.h"

namespace spomp {

/**
 * @class PanoPlanner
 * @brief Class for a panoramic planner
 */
    PanoPlanner::PanoPlanner(const Params& params) : params_(params) {
  auto& tm = TimerManager::getGlobal();
  pano_update_t_ = tm.get("PP");

  // Initialize assuming we have space around us
  // This way we are not paralyzed before getting a pano
  reachability_ = Reachability(0, {AngularProj::StartFinish{0, 2*pi}, 100});
  reachability_.getScan().setConstant(10);
}

/**
 * @brief Updates the panoramic traversability information based on a given TerrainPano object.
 *
 * @param pano The TerrainPano object containing depth and pose information.
 */
    void PanoPlanner::updatePano(const TerrainPano& pano) {
  pano_update_t_->start();

  // Zero for timestamp here is fine
  reachability_ = Reachability(0, pano.getAzProj(), pose32pose2(pano.getPose()));

  int gsize = params_.tbb <= 0 ? pano.cols() : params_.tbb;
  tbb::parallel_for(tbb::blocked_range<int>(0, reachability_.size(), gsize), 
    [&](tbb::blocked_range<int> range) {
      for (int col_i=range.begin(); col_i<range.end(); ++col_i) {
        // Move through column from bottom to top
        float last_r = 0;
        bool is_obs = false;
        for (int row_i=pano.rows()-1; row_i>=0; --row_i) {
          float r = pano.rangeAt(row_i, col_i);
          if (r > 0) {
            if (!pano.traversableAt(row_i, col_i)) {
              // We have hit an obstacle
              is_obs = true;
              break;
            }
            // If last_r is 0, not worth comparing
            if (r - last_r > params_.max_spacing_m && last_r > 0) {
              // We have hit a negative obstacle or large spacing
              break;
            }
            last_r = r;
          }
        }
        reachability_.setAzTrav(col_i, std::max<float>(last_r, 0.5), is_obs);
      }
    });
  
  pano_update_t_->end();
}

/**
 * @brief Plan a new goal point based on the given goal and old goal points.
 *
 * This function plans a new goal point based on the given goal and old goal points.
 *
 * @param[in] goal The current goal point to plan from.
 * @param[in] old_goal The previous goal point for consistency tracking.
 * @return The new goal point.
 */
    Eigen::Vector2f PanoPlanner::plan(const Eigen::Vector2f& goal,
                                      const Eigen::Vector2f& old_goal) const
{
  Eigen::Array2Xf samples(2, params_.sample_size);

  float max_range = reachability_.maxRange();

  if (max_range < 0.1) {
    // Basically nothing is free, just stay put
    return {0, 0};
  }

  // Sample feasible points
  static std::random_device rd; // Random seed
  static std::ranlux24_base gen(rd());
  std::uniform_real_distribution dis(-max_range, max_range);
  // This is actually faster not parallelized
  for (int sample_id=0; sample_id<samples.cols(); ++sample_id) {
    Eigen::Vector2f s;
    do {
      s = {dis(gen), dis(gen)};
    } while (!isSafe(s));
    samples.col(sample_id) = s;
  }

  Eigen::VectorXf dists = (samples.colwise() - goal.array()).matrix().colwise().norm();
  if (old_goal.norm() != 0) {
    // Penalize difference from the direction of the old goal
    // This encourages consistency in giving goals
    Eigen::Vector2f old_goal_norm = old_goal.normalized();
    // Cross product
    dists += params_.consistency_cost * (samples.row(0) * old_goal_norm[1] - 
        samples.row(1) * old_goal_norm[0]).abs().matrix();
  }
  int best_ind;
  dists.minCoeff(&best_ind);
  return samples.col(best_ind);
}

/**
* @brief Check if a point is safe to move to.
*
* This function determines whether a given 2D point is safe to move to.
* It checks the reachability of the position based on the existing scan data.
* If no scan has been performed yet, the function assumes everything is safe and returns true.
* If the point is at the origin, the function considers it safe and returns true.
* Otherwise, it converts the point from Cartesian to polar coordinates and checks if the
* range at the azimuth angle is greater than the distance from the origin to the point.
*
* @param pt The 2D point to check safety for.
* @return True if the point is safe, otherwise false.
*/
    bool PanoPlanner::isSafe(const Eigen::Vector2f& pt) const {
  if (reachability_.size() < 1) {
    // No scan yet, say everything is safe so we can move
    return true;
  }
  if (pt.norm() < 1e-5) {
    // Origin is safe
    return true;
  }
  Eigen::Vector2f polar = cart2polar(pt);
  return getRangeAtAz(polar[1]) > polar[0];
}

} // namespace spomp
