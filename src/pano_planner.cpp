#include <random>
#include <tbb/parallel_for.h>
#include "spomp/pano_planner.h"

namespace spomp {

PanoPlanner::PanoPlanner(const Params& params) : params_(params) {
  auto& tm = TimerManager::getGlobal();
  pano_update_t_ = tm.get("PP");
}

void PanoPlanner::updatePano(const TerrainPano& pano) {
  pano_update_t_->start();

  reachability_.scan = Eigen::VectorXf::Zero(pano.cols());
  reachability_.proj = pano.getAzProj();

  int gsize = params_.tbb <= 0 ? pano.cols() : params_.tbb;
  tbb::parallel_for(tbb::blocked_range<int>(0, pano.cols(), gsize), 
    [&](tbb::blocked_range<int> range) {
      for (int col_i=range.begin(); col_i<range.end(); ++col_i) {
        // Move through column from bottom to top
        float last_r = 0;
        for (int row_i=pano.rows()-1; row_i>=0; --row_i) {
          float r = pano.rangeAt(row_i, col_i);
          if (r > 0) {
            // If last_r is 0, not worth comparing
            if ((r - last_r > params_.max_spacing_m && last_r > 0) || 
                !pano.traversableAt(row_i, col_i)) 
            {
              // We have run into an obstacle or a large spacing, which
              // could be a negative obstacle
              break;
            }
            last_r = r;
          }
        }
        reachability_.scan[col_i] = last_r;
      }
    });
  
  pano_update_t_->end();
}

Eigen::Vector2f PanoPlanner::plan(const Eigen::Vector2f& goal, 
    const Eigen::Vector2f& old_goal) const 
{
  Eigen::Array3Xf samples(3, params_.sample_size);

  float max_range = reachability_.scan.maxCoeff();
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
    Eigen::Vector3f s;
    do {
      s = {dis(gen), dis(gen), 0};
    } while (!isSafe(s.head<2>()));
    samples.col(sample_id) = s;
  }

  Eigen::VectorXf dists = (samples.topRows<2>().colwise() - goal.array()).matrix().colwise().norm();
  if (old_goal.norm() != 0) {
    // Penalize difference from the direction of the old goal
    // This encourages consistency in giving goals
    Eigen::Vector3f old_goal3 = Eigen::Vector3f::Zero();
    old_goal3.head<2>()  = old_goal.normalized();
    dists += samples.colwise().cross(old_goal3).matrix().colwise().norm();
  }
  int best_ind;
  dists.minCoeff(&best_ind);
  return samples.col(best_ind).head<2>();
}

bool PanoPlanner::isSafe(const Eigen::Vector2f& pt) const {
  if (reachability_.scan.size() < 1) {
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
