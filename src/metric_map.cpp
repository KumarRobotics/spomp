#include "spomp/metric_map.h"

namespace spomp {

/**
* @brief Constructor for MetricMap class.
*
* This constructor initializes the MetricMap object using the provided parameters.
* It creates a grid map with the specified layers and sets the basic layers as "elevation" and "intensity".
* The frame ID is set to "map" and the geometry is initialized with a buffer size on all sides of (0, 0).
* The map is cleared.
* @param p The parameters for the MetricMap object.
*/
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

/**
 * @brief Adds a point cloud to the metric map.
 *
 * The point cloud is added to the map based on its position and intensity values.
 * The elevation and intensity layers of the map are updated with new values if necessary.
 * The semantics and semantics visualization layers are updated with the class information of the points in the cloud.
 *
 * @param cloud The point cloud to be added.
 * @param stamp The timestamp of the point cloud.
 */
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

/**
* @brief Clears the metric map.
*
* This function clears the metric map by setting all the layers to their default values.
* The timestamp of the metric map is also reset to 0.
*
* @param None
* @return None
*/
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

/**
 * @brief Resizes the MetricMap to fit within the specified bounds.
 *
 * This function resizes the MetricMap to fit within the specified minimum and maximum bounds.
 * The MetricMap is grown or shrunk as necessary to ensure that it fits within the specified bounds.
 * The MetricMap is centered at the current Position and has a size equal to the current Length.
 * The size of the MetricMap is increased by the buffer size defined in params_ if necessary.
 *
 * @param min The minimum bounds of the MetricMap.
 * @param max The maximum bounds of the MetricMap.
 */
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

/**
 * @brief Exports the metric map to a ROS message format.
 *
 * This function converts the internal grid_map::GridMap object to a grid_map_msgs::GridMap ROS message.
 * The timestamp of the grid_map::GridMap object is set to the value of most_recent_stamp_.
 *
 * @return The exported grid_map_msgs::GridMap ROS message.
 */
    grid_map_msgs::GridMap MetricMap::exportROSMsg() {
  grid_map_msgs::GridMap msg;
  map_.setTimestamp(most_recent_stamp_);
  grid_map::GridMapRosConverter::toMessage(map_, msg);
  return msg;
}

/**
 * @brief Checks if the metric map needs to be updated based on the given keyframe.
 *
 * This function compares the pose of the keyframe (frame) with the map pose of the keyframe.
 * If the translation distance or the rotation angle between the poses exceeds the threshold values,
 * the map needs to be updated.
 *
 * @param frame The keyframe to compare with.
 * @return True if the map needs to be updated, false otherwise.
 */
    bool MetricMap::needsMapUpdate(const Keyframe& frame) const {
  Eigen::Isometry3d diff = frame.getPose().inverse() * frame.getMapPose();
  return (diff.translation().norm() > params_.dist_for_rebuild_m ||
      Eigen::AngleAxisd(diff.rotation()).angle() > params_.ang_for_rebuild_rad);
}

/**
 * @brief Retrieve the counts of each class in the metric map.
 *
 * This function returns a vector of pointers to Eigen matrices, each representing the counts of a specific class in the metric map.
 *
 * @return A vector of pointers to Eigen matrices representing the counts of each class in the metric map.
 */
    std::vector<grid_map::Matrix*> MetricMap::getClsCounts() {
  std::vector<grid_map::Matrix*> counts;
  for (int cls=0; cls<class_config_.num_classes; ++cls) {
    counts.push_back(&map_[getClsCountLayerName(cls)]);
  }
  return counts;
}

} // namespace spomp
