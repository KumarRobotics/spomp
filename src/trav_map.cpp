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

Eigen::Vector2f TravMap::world2img(const Eigen::Vector2f& world_c) {
  return {0, 0};
}

Eigen::Vector2f TravMap::img2world(const Eigen::Vector2f& img_c) {
  return {0, 0};
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
  std::cout << max_terrain_ << std::endl;
  scaled_map.forEach<uint8_t>([&](uint8_t& p, const int* position) -> void {
    if (p == 0) {
      p = 255;
    } else {
      p = 255 * (p - 1) / max_terrain_;
    }
  });
  
  cv::Mat viz, cmapped_map;
  cv::applyColorMap(scaled_map, cmapped_map, cv::COLORMAP_PARULA);
  cmapped_map.copyTo(viz, map_ < 255);
  return viz;
}

} // namespace spomp
