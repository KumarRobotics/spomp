#include "spomp/metric_map.h"

namespace spomp {

MetricMap::MetricMap(const Params& p) : 
  params_(p),
  class_config_(semantics_manager::getClassesPath(params_.world_config_path)) 
{
  std::vector<std::string> layers{
      "elevation",
      "intensity",
      "semantics",
      "semantics_viz",
      "semantics2",
      "semantics2_viz"};
  for (int cls=0; cls<class_config_.num_classes; ++cls) {
    layers.push_back(getClsCountLayerName(cls));
  }
  map_ = grid_map::GridMap(layers);
  map_.setBasicLayers({"elevation", "intensity"});
  // Reset layers to the appropriate values
  map_.setFrameId("map");
  // Init with buffer on all sides of (0, 0)
  map_.setGeometry(grid_map::Length(params_.buffer_size_m, params_.buffer_size_m)*2,
      1/params_.resolution, grid_map::Position(0, 0));
  clear();
}

void MetricMap::addCloud(const PointCloudArray& cloud, long stamp) {
  if (stamp > most_recent_stamp_) {
    most_recent_stamp_ = stamp;
  }

  // Location in the world frame of the corner of the map (top-left, specifically)
  grid_map::Position corner_pos;
  map_.getPosition(grid_map::Index(0, 0), corner_pos);
  // Compute the indices in the map for each point in the cloud
  Eigen::Array2Xi inds = (((-cloud).topRows<2>().colwise() +
      corner_pos.array().cast<float>()) / map_.getResolution()).round().cast<int>();

  // Get references to map layers we need, speeds up access so we only do key lookup once
  grid_map::Matrix &elevation_layer = map_["elevation"];
  grid_map::Matrix &intensity_layer = map_["intensity"];
  grid_map::Matrix &semantics_layer = map_["semantics"];
  grid_map::Matrix &semantics_viz_layer = map_["semantics_viz"];
  grid_map::Matrix &semantics2_layer = map_["semantics2"];
  grid_map::Matrix &semantics2_viz_layer = map_["semantics2_viz"];
  auto class_counts = getClsCounts();

  grid_map::Index ind;
  for (int col=0; col<inds.cols(); col++) {
    ind = inds.col(col);
    if ((ind < 0).any() || (ind >= map_.getSize()).any()) {
      // Outside map bounds
      continue;
    }

    if (std::isnan(elevation_layer(ind[0], ind[1])) || 
        cloud(2, col) > elevation_layer(ind[0], ind[1])) 
    {
      elevation_layer(ind[0], ind[1]) = cloud(2, col);
      intensity_layer(ind[0], ind[1]) = cloud(3, col);
    }

    int sem_ind = cloud(4, col);
    // Don't overwrite with unknown
    if (sem_ind < class_config_.num_classes) {
      ++(*class_counts[sem_ind])(ind[0], ind[1]);
      if (sem_ind != semantics_layer(ind[0], ind[1])) {
        // Added new class not current max.  Recompute.
        int cls = -1;
        int max_cls = 0;
        int max_cnt = 0;
        for (const grid_map::Matrix* cnts : class_counts) {
          ++cls;
          if ((*cnts)(ind[0], ind[1]) > max_cnt) {
            max_cnt = (*cnts)(ind[0], ind[1]);
            max_cls = cls;
          }
        }

        cls = -1;
        int max_cls2 = 0;
        int max_cnt2 = 0;
        for (const grid_map::Matrix* cnts : class_counts) {
          ++cls;
          if (cls == max_cls) continue;
          if (class_config_.exclusivity[cls] && class_config_.exclusivity[max_cls]) continue;

          if ((*cnts)(ind[0], ind[1]) > max_cnt2) {
            max_cnt2 = (*cnts)(ind[0], ind[1]);
            max_cls2 = cls;
          }
        }

        semantics_layer(ind[0], ind[1]) = max_cls;
        uint32_t sem_color_packed = class_config_.color_lut.ind2Color(max_cls);
        semantics_viz_layer(ind[0], ind[1]) = *reinterpret_cast<float*>(&sem_color_packed);
        
        if (max_cnt2 > 5) {
          semantics2_layer(ind[0], ind[1]) = max_cls2;
          sem_color_packed = class_config_.color_lut.ind2Color(max_cls2);
          semantics2_viz_layer(ind[0], ind[1]) = *reinterpret_cast<float*>(&sem_color_packed);
        }
      }
    }
  }
}

void MetricMap::clear() {
  most_recent_stamp_ = 0;
  map_.setConstant("elevation", NAN);
  map_.setConstant("intensity", NAN);
  map_.setConstant("semantics", NAN);
  map_.setConstant("semantics_viz", 0);
  map_.setConstant("semantics2", NAN);
  map_.setConstant("semantics2_viz", 0);
  for (int cls=0; cls<class_config_.num_classes; ++cls) {
    map_.setConstant(getClsCountLayerName(cls), 0);
  }
}

void MetricMap::resizeToBounds(const Eigen::Vector2d& min, 
    const Eigen::Vector2d& max) 
{
  grid_map::Position center_pos = map_.getPosition();
  grid_map::Length size = map_.getLength();

  grid_map::Length diff = center_pos.array() - size/2 - min.array() + params_.buffer_size_m;
  if ((diff > 0).any()) {
    size += diff.cwiseMax(0);
    map_.grow(size, grid_map::GridMap::SW);
  }

  diff = -center_pos.array() - size/2 + max.array() + params_.buffer_size_m;
  if ((diff > 0).any()) {
    size += diff.cwiseMax(0);
    map_.grow(size, grid_map::GridMap::NE);
  }
}

grid_map_msgs::GridMap MetricMap::exportROSMsg() {
  grid_map_msgs::GridMap msg;
  map_.setTimestamp(most_recent_stamp_);
  grid_map::GridMapRosConverter::toMessage(map_, msg);
  return msg;
}

bool MetricMap::needsMapUpdate(const Keyframe& frame) const {
  Eigen::Isometry3d diff = frame.getPose().inverse() * frame.getMapPose();
  return (diff.translation().norm() > params_.dist_for_rebuild_m ||
      Eigen::AngleAxisd(diff.rotation()).angle() > params_.ang_for_rebuild_rad);
}

std::vector<grid_map::Matrix*> MetricMap::getClsCounts() {
  std::vector<grid_map::Matrix*> counts;
  for (int cls=0; cls<class_config_.num_classes; ++cls) {
    counts.push_back(&map_[getClsCountLayerName(cls)]);
  }
  return counts;
}

} // namespace spomp
