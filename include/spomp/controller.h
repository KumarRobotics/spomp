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
      float max_lin_vel = 1.5;
      float max_ang_vel = 1;
      float horizon_sec = 1;
      float horizon_dt = 0.1;
      int lin_disc = 10;
      int ang_disc = 20;
    };
    Controller(const Params& params);

    Twistf getControlInput(const Twistf& cur_vel, const Eigen::Isometry2f& state,
        const Eigen::Vector2f& goal, const PanoPlanner& planner) const;

    std::vector<Eigen::Isometry2f> forward(
        const Eigen::Isometry2f& state, const Twistf& vel) const;

    float trajCost(const std::vector<Eigen::Isometry2f>& traj,
        const Eigen::Vector2f& goal) const;

    bool isTrajSafe(const std::vector<Eigen::Isometry2f>& traj,
        const PanoPlanner& planner) const;

    static float angularDist(const Eigen::Isometry2f& pose,
        const Eigen::Vector2f& goal);

  protected:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;
};

} // namespace spomp
