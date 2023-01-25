#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p) : params_(p) {}

void AerialMap::updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf) {
  map_ = sem_map;
  map_ref_frame_ = mrf;
}

void AerialMap::updateLocalReachability(const Reachability& reach) {
}

float AerialMap::getEdgeProb(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2) const 
{
  return 0;
}

} // namespace spomp
