#pragma once

#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"

namespace spomp {

class TravMap {
  public:
    struct Params {
      std::string terrain_types_path = "";
    };
    TravMap(const Params& p);

    void updateMap(const cv::Mat& map, const Eigen::Vector2f& center);

    cv::Mat viz() const;

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void loadTerrainLUT();

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    cv::Mat terrain_lut_{};
    cv::Mat map_{};
    Eigen::Vector2f map_center_{};
    
    TravGraph graph_{};
};

} // namespace spomp
