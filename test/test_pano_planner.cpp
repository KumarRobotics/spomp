#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "spomp/pano_planner.h"
#include <iostream>

namespace spomp {

TEST(utils, test_cart_polar_conv) {
  Eigen::Vector2f x{1, 0};
  Eigen::Vector2f x_pol = cart2polar(x);
  ASSERT_FLOAT_EQ(x_pol[0], 1);
  ASSERT_FLOAT_EQ(x_pol[1], 0);

  Eigen::Vector2f x_back = polar2cart(x_pol);
  ASSERT_TRUE((x_back - x).norm() < 1e-5);
}

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
  ASSERT_EQ(pp.getReachability().is_obs[0], 1);

  Eigen::Vector2f goal{5, 0};
  Eigen::Vector2f local_goal = pp.plan(goal);
  ASSERT_TRUE(local_goal[0] < 3);
  ASSERT_TRUE(local_goal[1] < 3);
  Eigen::Vector2f local_goal2 = pp.plan(goal);
  ASSERT_TRUE(local_goal[0] != local_goal2[0]);

  Eigen::Vector2f goal2{1, 0};
  local_goal = pp.plan(goal2);
  // Technically this might sometimes not be true, but should be almost always
  ASSERT_TRUE((local_goal - goal2).norm() < 0.5);
}

static void BM_pano_planner_plan(benchmark::State& state) {
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
  Eigen::Vector2f goal{5, 0};
  for (auto _ : state) {
    Eigen::Vector2f lg = pp.plan(goal);
    benchmark::DoNotOptimize(lg);
  }
}
BENCHMARK(BM_pano_planner_plan);

} // namespace spomp
