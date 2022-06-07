#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "spomp/trav_map.h"

namespace spomp {

TravMap::TravMap(const Params& p) : params_(p) {
  loadTerrainLUT();
}

void TravMap::loadTerrainLUT() {
  terrain_lut_ = cv::Mat::ones(256, 1, CV_8UC1) * 255;
  if (params_.terrain_types_path != "") {
    const YAML::Node terrain_types = YAML::LoadFile(params_.terrain_types_path);
    int terrain_id = 0;
    for (const auto& terrain_type : terrain_types) {
      int type = terrain_type.as<int>();
      if (type >= 0) {
        // If less than 0, leave as untraversable
        terrain_lut_.at<uint8_t>(terrain_id) = type;
        if (type > max_terrain_) {
          max_terrain_ = type;
        }
      }
      ++terrain_id;
    }
  } else {
    // Default: assume first class is traversable, not second
    // Rest unknown
    terrain_lut_.at<uint8_t>(0) = 1;
    terrain_lut_.at<uint8_t>(1) = 0;
  }
}

Eigen::Vector2f TravMap::world2img(const Eigen::Vector2f& world_c) const {
  Eigen::Vector2f world_pt = world_c - map_center_;
  Eigen::Vector2f img_pt = {-world_pt[1], -world_pt[0]};
  img_pt *= params_.map_res;
  img_pt += Eigen::Vector2f(map_.cols, map_.rows)/2;
  return img_pt;
}

Eigen::Vector2f TravMap::img2world(const Eigen::Vector2f& img_c) const {
  Eigen::Vector2f img_pt = img_c - Eigen::Vector2f(map_.cols, map_.rows)/2;
  img_pt /= params_.map_res;
  Eigen::Vector2f world_pt = {-img_pt[1], -img_pt[0]};
  world_pt += map_center_;
  return world_pt;
}

void TravMap::updateMap(const cv::Mat &map, const Eigen::Vector2f& center) {
  if (map.channels() == 1) {
    map_ = map;
  } else {
    // In the event we receive a 3 channel image, assume all channels are
    // the same
    cv::cvtColor(map, map_, cv::COLOR_BGR2GRAY);
  }
  cv::LUT(map_, terrain_lut_, map_);
  map_center_ = center;
}

cv::Mat TravMap::viz() const {
  cv::Mat scaled_map = map_.clone();
  // Rescale for increasing order of trav difficulty 0->255
  scaled_map.forEach<uint8_t>([&](uint8_t& p, const int* position) -> void {
    if (p == 0) {
      p = 255;
    } else {
      p = 255 * (p - 1) / max_terrain_;
    }
  });
  
  cv::Mat viz, cmapped_map;
  cv::applyColorMap(scaled_map, cmapped_map, cv::COLORMAP_PARULA);
  // Mask unknown regions
  cmapped_map.copyTo(viz, map_ < 255);
  // Draw origin
  auto origin_img = world2img({0, 0});
  cv::circle(viz, cv::Point(origin_img[0], origin_img[1]), 3, cv::Scalar(0, 0, 255), 2);
  return viz;
}

} // namespace spomp
