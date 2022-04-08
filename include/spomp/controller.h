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
      float max_ang_accel = 1;
      float max_lin_vel = 1.5;
      float max_ang_vel = 1;
      float horizon_sec = 1;
      float horizon_dt = 0.1;
      int lin_disc = 10;
      int ang_disc = 20;
      float obs_cost_weight = 1;
      Eigen::Isometry3f control_trans = Eigen::Isometry3f::Identity();
    };
    Controller(const Params& params);

    void setGoal(const Eigen::Vector2f& goal) {
      goal_ = goal;
    }

    const auto& getLocalGoal() const {
      return goal_;
    }

    /*!
     * Get the best control input given a map and goal
     * @param cur_vel Current velocity of the robot
     * @param state_p Current 2D projection of robot pose
     * @param goal Goal point in pano frame
     * @param planner Reference to planner for obstacle avoidance
     * @param pano Reference to terrain pano
     */
    Twistf getControlInput(const Twistf& cur_vel, const Eigen::Isometry3f& state_p,
        const PanoPlanner& planner, const TerrainPano& pano) const;

    //! Forward simulate a velocity into a trajectory
    std::vector<Eigen::Isometry2f> forwardFromOrigin(const Twistf& vel) const {
      return forward(Eigen::Isometry2f::Identity(), vel);
    };

  protected:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    //! Forward simulate a velocity into a trajectory
    std::vector<Eigen::Isometry2f> forward(
        const Eigen::Isometry2f& state, const Twistf& vel) const;

    //! Compute the trajectory cost (ignoring obstacles)
    float trajCostGoal(const std::vector<Eigen::Isometry2f>& traj) const;

    //! Compute the trajectory cost (ignoring obstacles)
    float trajCostObs(const std::vector<Eigen::Isometry2f>& traj,
        const TerrainPano& pano) const;

    //! Return true if trajectory is collision free
    bool isTrajSafe(const std::vector<Eigen::Isometry2f>& traj,
        const PanoPlanner& planner) const;

    /*! 
     * Returns the angular difference between the pose orientation
     * and the vector from the current location to the goal.  Used to encourage
     * robot to face towards goal
     */
    static float angularDist(const Eigen::Isometry2f& pose,
        const Eigen::Vector2f& goal);

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    Eigen::Vector2f goal_{Eigen::Vector2f::Zero()};

    // Timers
    Timer* controller_t_;
};

} // namespace spomp
