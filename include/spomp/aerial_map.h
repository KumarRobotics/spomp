#pragma once

#include <dlib/dnn.h>
#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"

namespace spomp {

class AerialMap {
  public:
    struct Params {
    };
    AerialMap(const Params& p);

    void updateMap(const cv::Mat& sem_map, const cv::Mat& other_channels);

    float getEdgeProb(const Eigen::Vector2f& n1, const Eigen::Vector2f& n2) const;

    void testML();

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
};

} // namespace spomp
