#pragma once

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>

namespace spomp {

class MetricMap {
  public:
    struct Params {
    };
    MetricMap(const Params& p);

  private:
    Params params_{};

    grid_map::GridMap map_;
};

} // namespace spomp
