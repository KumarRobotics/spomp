#include <opencv2/imgproc/imgproc.hpp>
#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p, const MLPModel::Params& mlp_p) : 
  params_(p), model_(mlp_p) {}

void AerialMap::updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf) {
  sem_map_ = sem_map;
  map_ref_frame_ = mrf;
  trav_map_ = cv::Mat::zeros(sem_map.size(), CV_16SC1);
}

void AerialMap::updateLocalReachability(const Reachability& reach) {
  Eigen::Vector2f img_center = 
    map_ref_frame_.world2img(reach.getPose().translation());

  Eigen::VectorXf thetas = reach.getProj().getAngles();
  thetas.array() += Eigen::Rotation2Df(reach.getPose().rotation()).angle();

  cv::Mat trav_map_delta = cv::Mat::zeros(trav_map_.size(), CV_16SC1);

  for (int i=0; i<thetas.size(); ++i) {
    Eigen::Vector2f ray_dir(cos(thetas[i]), sin(thetas[i]));
    auto ray_info = reach.getObsAtInd(i);
    for (float range=0; range<ray_info.range; range+=1./map_ref_frame_.res) {
      Eigen::Vector2f pt = ray_dir*range + img_center;
      if (map_ref_frame_.imgPointInMap(pt)) {
        trav_map_delta.at<int16_t>(cv::Point(pt[0], pt[1])) = 1;
      }
    }

    if (ray_info.is_obs) {
      for (float range=ray_info.range; range<ray_info.range + params_.not_trav_range_m; 
          range+=1./map_ref_frame_.res) 
      {
        Eigen::Vector2f pt = ray_dir*range + img_center;
        if (map_ref_frame_.imgPointInMap(pt)) {
          trav_map_delta.at<int16_t>(cv::Point(pt[0], pt[1])) = -1;
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

Eigen::VectorXf AerialMap::getFeatureAtPoint(const cv::Point& pt) {
  return {};
}

void AerialMap::fitModel() {
  // No more features than number of nonzero pixels in trav map
  int max_features = cv::countNonZero(trav_map_);
  Eigen::ArrayXXf features(6, max_features);
  Eigen::VectorXi labels(max_features);

  int n_features = 0;
  for(int i=0; i<trav_map_.rows; i++) {
    for(int j=0; j<trav_map_.cols; j++) {
      if (trav_map_.at<int16_t>(i,j) <= -params_.not_trav_thresh) {
        features.col(n_features) = getFeatureAtPoint(cv::Point(i,j));
        labels[n_features] = 0;
        ++n_features;
      } else if (trav_map_.at<int16_t>(i,j) >= params_.trav_thresh) {
        features.col(n_features) = getFeatureAtPoint(cv::Point(i,j));
        labels[n_features] = 1;
        ++n_features;
      }
    }
  }

  model_.fit(features, labels);
  updateProbabilityMap();
}

void AerialMap::updateProbabilityMap() {
}

cv::Mat AerialMap::viz() {
  cv::Mat trav_viz;
  constexpr int new_center = std::numeric_limits<uint8_t>::max()/2;
  trav_map_.convertTo(trav_viz, CV_8UC1, 1, new_center);

  auto center = map_ref_frame_.world2img({0,0});

  cv::cvtColor(trav_viz, trav_viz, cv::COLOR_GRAY2BGR);

  using Pixel = cv::Point3_<uint8_t>;
  trav_viz.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
    if (pixel.x <= new_center - params_.not_trav_thresh) {
      pixel.x = 0;   // b
      pixel.y = 0;   // g
      pixel.z = 255; // r
    } else if (pixel.x >= new_center + params_.trav_thresh) {
      pixel.x = 0;   // b
      pixel.y = 255; // g
      pixel.z = 0;   // r
    } else {
      // white
      pixel.x = 255;
      pixel.y = 255;
      pixel.z = 255;
    }
  });

  return trav_viz;
}

} // namespace spomp
