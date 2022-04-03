#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "spomp/pano_planner.h"
#include <iostream>

namespace spomp {

TEST(pano_planner, test) {
  PanoPlanner pp({});  
  TerrainPano tp({});

  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Zero(256, 1024);
  Eigen::VectorXf alts = Eigen::VectorXf::LinSpaced(pano.rows(), 
      deg2rad(90)/2, -deg2rad(90)/2);
  for (int col=0; col<pano.cols(); ++col) {
    pano.col(col) = 0.4 / (-alts).array().sin().cwiseMax(0.4/3);
  }

  tp.updatePano(pano, {});
  pp.updatePano(tp);

  ASSERT_TRUE(pp.getReachability().scan[0] > 2);
  ASSERT_TRUE(pp.getReachability().scan[0] < 3);
}

} // namespace spomp
