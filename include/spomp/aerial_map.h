#pragma once

#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"
#include "spomp/mlp_model.h"

namespace spomp {

class AerialMap {
  public:
    struct Params {
      int trav_thresh = 1;
      int not_trav_thresh = 1;
      float not_trav_range_m = 3;
    };
    AerialMap(const Params& p, const MLPModel::Params& mlp_p);

    void updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf);

    void updateLocalReachability(const Reachability& reach);

    float getEdgeProb(const Eigen::Vector2f& n1, const Eigen::Vector2f& n2) const;

    void fitModel();

    cv::Mat viz();

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    Eigen::VectorXf getFeatureAtPoint(const cv::Point& pt);
    void updateProbabilityMap();

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    cv::Mat sem_map_{};
    MapReferenceFrame map_ref_frame_{};

    cv::Mat trav_map_{};
    cv::Mat prob_map_{};

    MLPModel model_;
};

} // namespace spomp
