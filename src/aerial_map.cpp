#include <opencv2/imgproc/imgproc.hpp>
#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p, const MLPModel::Params& mlp_p) : 
  params_(p), model_(mlp_p) {}

void AerialMap::updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf) {
  if (sem_map.channels() == 1) {
    sem_map_ = sem_map;
  } else {
    // In the event we receive a 3 channel image, assume all channels are
    // the same
    cv::cvtColor(sem_map, sem_map_, cv::COLOR_BGR2GRAY);
  }
  map_ref_frame_ = mrf;
  trav_map_ = cv::Mat::zeros(sem_map.size(), CV_16SC1);
  prob_map_ = cv::Mat::zeros(sem_map.size(), CV_32FC1);
}

void AerialMap::updateLocalReachability(const Reachability& reach) {
  Eigen::Vector2f img_center = 
    map_ref_frame_.world2img(reach.getPose().translation());

  Eigen::VectorXf thetas = reach.getProj().getAngles();
  thetas.array() += Eigen::Rotation2Df(reach.getPose().rotation()).angle();

  cv::Mat trav_map_delta = cv::Mat::zeros(trav_map_.size(), CV_16SC1);

  for (int i=0; i<thetas.size(); ++i) {
    Eigen::Vector2f ray_dir(-sin(thetas[i]), -cos(thetas[i]));
    auto ray_info = reach.getObsAtInd(i);
    for (float range=0; range<ray_info.range; range+=1./map_ref_frame_.res) {
      Eigen::Vector2f pt = ray_dir*range*map_ref_frame_.res + img_center;
      if (map_ref_frame_.imgPointInMap(pt)) {
        trav_map_delta.at<int16_t>(cv::Point(pt[0], pt[1])) = 1;
      }
    }

    if (ray_info.is_obs) {
      for (float range=ray_info.range; range<ray_info.range + params_.not_trav_range_m; 
          range+=1./map_ref_frame_.res) 
      {
        Eigen::Vector2f pt = ray_dir*range*map_ref_frame_.res + img_center;
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
  Eigen::VectorXf feat = Eigen::VectorXf::Zero(6);
  int cls = sem_map_.at<uint8_t>(pt);
  if (cls < 6) {
    feat[cls] = 1;
  }
  return feat;
}

void AerialMap::fitModel() {
  // No more features than number of nonzero pixels in trav map
  int max_features = cv::countNonZero(trav_map_);
  if (max_features == 0) return;
  Eigen::ArrayXXf features(6, max_features);
  Eigen::VectorXi labels(max_features);

  int n_features = 0;
  for(int i=0; i<trav_map_.rows; ++i) {
    for(int j=0; j<trav_map_.cols; ++j) {
      cv::Point pt(j, i);
      if (trav_map_.at<int16_t>(pt) <= -params_.not_trav_thresh) {
        features.col(n_features) = getFeatureAtPoint(pt);
        labels[n_features] = 0;
        ++n_features;
      } else if (trav_map_.at<int16_t>(pt) >= params_.trav_thresh) {
        features.col(n_features) = getFeatureAtPoint(pt);
        labels[n_features] = 1;
        ++n_features;
      }
    }
  }
  features.conservativeResize(Eigen::NoChange, n_features);
  labels.conservativeResize(n_features);

  model_.fit(features, labels);
  updateProbabilityMap();
}

void AerialMap::updateProbabilityMap() {
  int n_features = cv::countNonZero(sem_map_ < 255);
  Eigen::ArrayXXf features(6, n_features);
  std::vector<float*> prob_ptrs(n_features);

  int feat_cnt = 0;
  for(int i=0; i<prob_map_.rows; ++i) {
    for(int j=0; j<prob_map_.cols; ++j) {
      cv::Point pt(j, i);

      if (sem_map_.at<uint8_t>(pt) < 255) {
        prob_ptrs[feat_cnt] = prob_map_.ptr<float>(i, j);
        features.col(feat_cnt) = getFeatureAtPoint(pt);
        ++feat_cnt;
      }
    }
  }

  Eigen::VectorXf pred_log_probs = model_.infer(features);
  for (int i=0; i<feat_cnt; ++i) {
    *(prob_ptrs[i]) = std::exp(pred_log_probs[i]);
  }
}

cv::Mat AerialMap::viz() const {
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
      // black
      pixel.x = 0;
      pixel.y = 0;
      pixel.z = 0;
    }
  });

  cv::Mat prob_viz;
  prob_map_.convertTo(prob_viz, CV_8UC1, 255, 0);
  cv::applyColorMap(prob_viz, prob_viz, cv::COLORMAP_RAINBOW);
  cv::addWeighted(trav_viz, 1, prob_viz, 0.5, 0, trav_viz);

  return trav_viz;
}

} // namespace spomp
