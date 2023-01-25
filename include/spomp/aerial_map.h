#pragma once

#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"
#include "spomp/mlp_model.h"

namespace spomp {

class AerialMap {
  public:
    struct Params {
    };
    AerialMap(const Params& p);

    void updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf);

    void updateLocalReachability(const Reachability& reach);

    float getEdgeProb(const Eigen::Vector2f& n1, const Eigen::Vector2f& n2) const;

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    cv::Mat map_{};
    MapReferenceFrame map_ref_frame_{};
};

} // namespace spomp
