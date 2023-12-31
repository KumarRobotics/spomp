#pragma once

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include "spomp/keyframe.h"
#include "semantics_manager/semantics_manager.h"

namespace spomp {

class MetricMap {
  public:
    struct Params {
      std::string world_config_path = "";

      float resolution = 2; // In cell/m
      float buffer_size_m = 50;
      float req_point_density = 500;

      float dist_for_rebuild_m = 3;
      float ang_for_rebuild_rad = 0.1;
    };
    MetricMap(const Params& p);

    void addCloud(const PointCloudArray& cloud, long stamp);

    void clear();

    void resizeToBounds(const Eigen::Vector2d& min, const Eigen::Vector2d& max);

    grid_map_msgs::GridMap exportROSMsg();

    bool needsMapUpdate(const Keyframe& frame) const;

    std::vector<grid_map::Matrix*> getClsCounts();

    static std::string getClsCountLayerName(int cls_id) {
      return "cls_cnt" + std::to_string(cls_id);
    }

    //! This is really just for test purposes
    const auto& getMap() const {
      return map_;
    }

  private:
    Params params_{};

    semantics_manager::ClassConfig class_config_;

    grid_map::GridMap map_{};

    long most_recent_stamp_{0};
};

} // namespace spomp
