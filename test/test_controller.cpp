#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include "spomp/controller.h"

namespace spomp {

TEST(controller, test_forward) {
  Controller c({});

  auto traj = c.forward(Eigen::Isometry2f::Identity(), Twistf(2*pi, 2*pi));
  // This should be going in a circle around (0, 1)
  for (const auto& pos : traj) {
    EXPECT_NEAR((pos.translation() - Eigen::Vector2f(0, 1)).norm(), 1, 1e-5);
  }
  EXPECT_NEAR((traj[5].translation() - Eigen::Vector2f(0, 2)).norm(), 0, 1e-5);
  EXPECT_NEAR((traj[10].translation() - Eigen::Vector2f(0, 0)).norm(), 0, 1e-5);

  Eigen::Isometry2f init = Eigen::Isometry2f::Identity();
  init.translate(Eigen::Vector2f(1, 0));
  init.rotate(Eigen::Rotation2Df(deg2rad(-90)));
  traj = c.forward(init, Twistf(2*pi, -2*pi));
  // CW circle around (0, 0)
  EXPECT_NEAR((traj[5].translation() - Eigen::Vector2f(-1, 0)).norm(), 0, 1e-5);

  // Linear motion
  traj = c.forward(Eigen::Isometry2f::Identity(), Twistf(10, 0));
  EXPECT_NEAR((traj[10].translation() - Eigen::Vector2f(10, 0)).norm(), 0, 1e-5);
}

TEST(controller, test_traj_cost) {
  Controller c({});

  std::vector<Eigen::Isometry2f> traj;
  Eigen::Isometry2f pose = Eigen::Isometry2f::Identity();
  traj.push_back(pose);
  pose.translate(Eigen::Vector2f(1, 0));
  pose.rotate(Eigen::Rotation2Df(pi/2));
  traj.push_back(pose);

  EXPECT_FLOAT_EQ(c.trajCost(traj, {1, 0}), 0);
  EXPECT_FLOAT_EQ(c.trajCost(traj, {1, 1}), 1);
  EXPECT_FLOAT_EQ(c.trajCost(traj, {1, -1}), 1+pi/10);
}

TEST(controller, test_get_control_input) {
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
  Controller c({});

  auto twist = c.getControlInput({}, Eigen::Isometry2f::Identity(), {5, 0}, pp);
  EXPECT_FLOAT_EQ(twist.linear(), 0.1);
  // Not exact because of input disc
  EXPECT_NEAR(twist.ang(), 0, 1e-3);

  twist = c.getControlInput({}, Eigen::Isometry2f::Identity(), {-5, 1}, pp);
  // Slight turn in place
  EXPECT_FLOAT_EQ(twist.linear(), 0);
  EXPECT_FLOAT_EQ(twist.ang(), 0.01);

  twist = c.getControlInput({1, 0}, Eigen::Isometry2f::Identity(), {-5, 1}, pp);
  // Slight turn in place
  EXPECT_FLOAT_EQ(twist.linear(), 0.9);
  EXPECT_FLOAT_EQ(twist.ang(), 0.01);
}

static void BM_controller(benchmark::State& state) {
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
  Controller c({});

  for (auto _ : state) {
    c.getControlInput({}, Eigen::Isometry2f::Identity(), {5, 0}, pp);
  }
}
BENCHMARK(BM_controller);
  
} // namespace spomp
