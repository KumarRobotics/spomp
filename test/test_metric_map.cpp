#include <gtest/gtest.h>
#include <ros/package.h>

#include "spomp/metric_map.h"

namespace spomp {

TEST(metric_map, test_map) {
  MetricMap::Params mm_p;
  mm_p.world_config_path = 
    ros::package::getPath("semantics_manager") + "/config/test_config.yaml";
  mm_p.resolution = 10;
  mm_p.buffer_size_m = 25;
  mm_p.req_point_density = 0;
  MetricMap map(mm_p);

  // Check initial shape of map
  grid_map::Position pos;
  map.getMap().getPosition(grid_map::Index(0, 0), pos);
  EXPECT_FLOAT_EQ(pos[0], 50./2 - 0.1/2);
  EXPECT_FLOAT_EQ(pos[1], 50./2 - 0.1/2);
  map.getMap().getPosition(map.getMap().getSize() - 1, pos);
  EXPECT_FLOAT_EQ(pos[0], -50./2 + 0.1/2);
  EXPECT_FLOAT_EQ(pos[1], -50./2 + 0.1/2);

  PointCloudArray cloud(5, 2);
  grid_map::Position pos1(0.05, 0.05);
  grid_map::Position pos2(5.05, 0.05);
  cloud.col(0) << pos1.cast<float>(), 1, 1, 1;
  cloud.col(1) << pos2.cast<float>(), 2, 2, 2;
  map.addCloud(cloud, 0);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos1), 1);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos2), 2);
  EXPECT_EQ(map.getMap().atPosition("intensity", pos1), 1);
  EXPECT_EQ(map.getMap().atPosition("intensity", pos2), 2);

  pos1 = grid_map::Position(0.01, 0.09);
  pos2 = grid_map::Position(5.01, 0.09);
  cloud.col(0) << pos1.cast<float>(), 1, 2, 2;
  cloud.col(1) << pos2.cast<float>(), 3, 3, 3;
  map.addCloud(cloud, 0);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos1), 1);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos2), 3);
  EXPECT_EQ(map.getMap().atPosition("intensity", pos1), 1);
  EXPECT_EQ(map.getMap().atPosition("intensity", pos2), 3);

  pos1 = grid_map::Position(0.0, 0.0);
  cloud.col(0) << pos1.cast<float>(), 1, 2, 2;
  map.addCloud(cloud, 0);
  EXPECT_EQ(map.getMap().atPosition("intensity", pos1), 2);

  cloud.col(0) << pos1.cast<float>(), 0, 4, 4;
  map.addCloud(cloud, 0);
  EXPECT_EQ(map.getMap().atPosition("intensity", pos1), 2);
}

} // namespace spomp
