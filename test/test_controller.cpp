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
  
} // namespace spomp
