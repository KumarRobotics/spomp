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

    void updateMap(const cv::Mat& map);

    cv::Mat viz();

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
    
    TravGraph graph_{};
};

} // namespace spomp
