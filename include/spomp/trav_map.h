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
    Eigen::Vector2f world2img(const Eigen::Vector2f& world_c);
    Eigen::Vector2f img2world(const Eigen::Vector2f& img_c);

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    int max_terrain_{1};
    cv::Mat terrain_lut_{};
    cv::Mat map_{};
    Eigen::Vector2f map_center_{};
    
    TravGraph graph_{};
};

} // namespace spomp
