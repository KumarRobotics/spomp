#pragma once

#include <Eigen/Dense>
#include "spomp/utils.h"
#include "spomp/pano_planner.h"

namespace spomp {

class Controller {
  public:
    struct Params {
      float freq = 10;
      float max_lin_accel = 1;
      float max_ang_accel = 0.1;
      float horizon_sec = 1;
      // Best if these are odd, so stationary is an option
      int lin_disc = 11;
      int ang_disc = 21;
    };
    Controller(const Params& params);

    Twistf getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
      const PanoPlanner& pano);
    std::vector<Eigen::Isometry2f> forward(
        const Eigen::Isometry2f& state, const Twistf& vel);
    float scoreTraj(const std::vector<Eigen::Isometry2f>& traj);
    bool isTrajSafe(const std::vector<Eigen::Isometry2f>& traj);

  protected:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;
};

} // namespace spomp
