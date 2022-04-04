#pragma once

#include "spomp/terrain_pano.h"
#include "spomp/pano_planner.h"
#include "spomp/controller.h"

namespace spomp {

class Local {
  public:
    Local(const TerrainPano::Params& tp_p, const PanoPlanner::Params& pp_p,
        const Controller::Params& c_p);

    void updatePano(const Eigen::ArrayXXf& pano, const Eigen::Isometry3f& pose);

    Twistf getControlInput(const Eigen::Isometry2f& state);

    //! @param goal The goal in the odom frame
    void setGoal(const Eigen::Vector3f& goal);

    const TerrainPano& getPano() const {
      return pano_;
    }

    const PanoPlanner& getPlanner() const {
      return planner_;
    }

    const Controller& getController() const {
      return controller_;
    }

  protected:
    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    TerrainPano pano_;
    PanoPlanner planner_;
    Controller controller_;

    Twistf cur_vel_{};

    Eigen::Vector3f global_goal_{Eigen::Vector3f::Zero()};
};

} // namespace spomp
