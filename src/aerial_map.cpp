#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p, const MLPModel::Params& mlp_p) : 
  params_(p), model_(mlp_p) {}

void AerialMap::updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf) {
  sem_map_ = sem_map;
  map_ref_frame_ = mrf;
}

void AerialMap::updateLocalReachability(const Reachability& reach) {
  Eigen::Vector2f img_center = 
    map_ref_frame_.world2img(reach.getPose().translation());

  Eigen::VectorXf thetas = reach.getProj().getAngles();
  thetas.array() += Eigen::Rotation2Df(reach.getPose().rotation()).angle();

  for (int i=0; i<thetas.size(); ++i) {
    Eigen::Vector2f ray_dir(cos(thetas[i]), sin(thetas[i]));
    auto ray_info = reach.getObsAtInd(i);
    for (float range=0; range<ray_info.range; range+=map_ref_frame_.res) {
      Eigen::Vector2f pt = ray_dir*range + img_center;
    }
    if (ray_info.is_obs) {
    }
  }
}

float AerialMap::getEdgeProb(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2) const 
{
  return 0;
}

void AerialMap::fitModel() {
}

cv::Mat AerialMap::viz() {
  return {};
}

} // namespace spomp
