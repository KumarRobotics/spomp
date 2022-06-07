#include <yaml-cpp/yaml.h>
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
      if (terrain_type.as<int>() >= 0) {
        // If less than 0, leave as untraversable
        terrain_lut_.at<uint8_t>(terrain_id) = terrain_type.as<int>();
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

void TravMap::updateMap(const cv::Mat &map, const Eigen::Vector2f& center) {
  map_ = map;
  map_center_ = center;
}

cv::Mat TravMap::viz() const {
  return {};
}

} // namespace spomp
