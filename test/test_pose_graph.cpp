#include <gtest/gtest.h>
#include <iostream>

#include "spomp/utils.h"
#include "spomp/pose_graph.h"

namespace spomp {

TEST(pose_graph, test_init) {
  PoseGraph::Params pg_p;
  pg_p.num_frames_opt = 0;
  PoseGraph pg(pg_p);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translate(Eigen::Vector3d(1,0,0));
  pose.rotate(Eigen::AngleAxisd(pi/2, Eigen::Vector3d::UnitZ()));
  pg.addNode(1, pose);
  auto opt_pose = pg.getPoseAtTime(1);
  ASSERT_TRUE(opt_pose);
  EXPECT_EQ(opt_pose->translation().norm(), 0);

  // Translate in the current frame, so if the old frame is at origin,
  // new location will be at this position
  pose.translate(Eigen::Vector3d(1,0,0));
  auto node_id = pg.addNode(2, pose);
  EXPECT_FLOAT_EQ(pg.getPoseAtIndex(node_id).translation()[0], 1);
  EXPECT_FLOAT_EQ(pg.getPoseAtIndex(node_id).translation().norm(), 1);
  EXPECT_THROW({
      pg.getPoseAtIndex(3);
    }, std::out_of_range);
}

TEST(pose_graph, test_opt) {
  PoseGraph::Params pg_p;
  pg_p.num_frames_opt = 0;
  PoseGraph pg(pg_p);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pg.addNode(1, pose);
  pose.translate(Eigen::Vector3d(1,0,0));
  pg.addNode(2, pose);
  pose.translate(Eigen::Vector3d(1,0,0));
  pg.addNode(3, pose);

  ASSERT_TRUE(pg.getError() == 0);
  ASSERT_FLOAT_EQ(pg.getPoseAtTime(3)->translation().norm(), 2);

  Eigen::Isometry2d prior = Eigen::Isometry2d::Identity();
  prior.translate(Eigen::Vector2d(0,1));
  prior.rotate(Eigen::Rotation2D(pi/2));
  pg.addPrior(1, {prior});
  ASSERT_TRUE(pg.getError() > 0.001);
  pg.update();
  ASSERT_TRUE(pg.getError() < 0.001);
  ASSERT_NEAR(pg.getPoseAtTime(3)->translation()[0], 0, 1e-5);
  ASSERT_NEAR(pg.getPoseAtTime(3)->translation()[1], 3, 1e-5);

  pg.addPrior(3, {prior});
  auto error = pg.getError();
  pg.update();
  ASSERT_TRUE(pg.getError() < error);
  ASSERT_TRUE(pg.getError() > 1);
}

} // namespace spomp
