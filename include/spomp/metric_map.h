#pragma once

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include "spomp/keyframe.h"

namespace spomp {

class MetricMap {
  public:
    struct Params {
      float resolution = 2; // In cell/m
      float buffer_size_m = 50;
      float req_point_density = 500;
    };
    MetricMap(const Params& p);

    void addCloud(const PointCloudArray& cloud, long stamp);

    void clear();

    void resizeToBounds(const Eigen::Vector2d& min, const Eigen::Vector2d& max);

    auto exportROSMsg();

    //! This is really just for test purposes
    const auto& getMap() const {
      return map_;
    }

  private:
    Params params_{};

    grid_map::GridMap map_{};

    long most_recent_stamp_{0};
};

} // namespace spomp
