#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p, const MLPModel::Params& mlp_p) : 
  params_(p), model_(mlp_p) {}

void AerialMap::updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf) {
  sem_map_ = sem_map;
  map_ref_frame_ = mrf;
  trav_map_ = cv::Mat::zeros(trav_map_.size(), CV_32SC1);
}

void AerialMap::updateLocalReachability(const Reachability& reach) {
  Eigen::Vector2f img_center = 
    map_ref_frame_.world2img(reach.getPose().translation());

  Eigen::VectorXf thetas = reach.getProj().getAngles();
  thetas.array() += Eigen::Rotation2Df(reach.getPose().rotation()).angle();

  cv::Mat trav_map_delta = cv::Mat::zeros(trav_map_.size(), CV_32SC1);

  for (int i=0; i<thetas.size(); ++i) {
    Eigen::Vector2f ray_dir(cos(thetas[i]), sin(thetas[i]));
    auto ray_info = reach.getObsAtInd(i);
    for (float range=0; range<ray_info.range; range+=map_ref_frame_.res) {
      Eigen::Vector2f pt = ray_dir*range + img_center;
      if (map_ref_frame_.imgPointInMap(pt)) {
        trav_map_delta.at<int32_t>(cv::Point(pt[0], pt[1])) = 1;
      }
    }

    if (ray_info.is_obs) {
      for (float range=ray_info.range; range<ray_info.range+3; range+=map_ref_frame_.res) {
        Eigen::Vector2f pt = ray_dir*range + img_center;
        if (map_ref_frame_.imgPointInMap(pt)) {
          trav_map_delta.at<int32_t>(cv::Point(pt[0], pt[1])) = -1;
        }
      }
    }
  }

  trav_map_ += trav_map_delta;
}

float AerialMap::getEdgeProb(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2) const 
{
  return 0;
}

void AerialMap::fitModel() {
}

cv::Mat AerialMap::viz() {
  cv::Mat trav_viz;
  cv::cvtColor(trav_map_, trav_viz, cv::COLOR_GRAY2BGR);
  return {};
}

} // namespace spomp
