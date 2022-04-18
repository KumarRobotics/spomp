#pragma once

#include "spomp/terrain_pano.h"
#include "spomp/pano_planner.h"
#include "spomp/controller.h"

namespace spomp {

class Local {
  public:
    struct Params {
      float goal_thresh_m = 2;
    };
    Local(const Local::Params& l_p, const TerrainPano::Params& tp_p, 
        const PanoPlanner::Params& pp_p, const Controller::Params& c_p);

    void updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose);

    //! @param state The robot pose in the odom frame
    Twistf getControlInput(const Eigen::Isometry3f& state);

    //! @param goal The goal in the odom frame
    void setGoal(const Eigen::Vector3f& goal);

    const auto& getPano() const {
      return pano_;
    }

    const auto& getPlanner() const {
      return planner_;
    }

    const auto& getController() const {
      return controller_;
    }

    const auto& getGlobalGoal() const {
      return global_goal_;
    }

  protected:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    TerrainPano pano_;
    PanoPlanner planner_;
    Controller controller_;

    Twistf cur_vel_{};

    std::optional<Eigen::Vector3f> global_goal_{};
};

} // namespace spomp
