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
  pg.addNode(10, pose);

  ASSERT_TRUE(pg.getError() == 0);
  ASSERT_FLOAT_EQ(pg.getPoseAtTime(10)->translation().norm(), 2);

  Eigen::Isometry2d prior = Eigen::Isometry2d::Identity();
  prior.translate(Eigen::Vector2d(0,1));
  prior.rotate(Eigen::Rotation2D(pi/2));
  pg.addPrior(1, {prior});
  ASSERT_TRUE(pg.getError() > 0.001);
  pg.update();
  ASSERT_TRUE(pg.getError() < 0.001);
  ASSERT_NEAR(pg.getPoseAtTime(10)->translation()[0], 0, 1e-5);
  ASSERT_NEAR(pg.getPoseAtTime(10)->translation()[1], 3, 1e-5);

  pg.addPrior(9, {prior});
  auto error = pg.getError();
  pg.update();
  // Verify that nothing happens here because disallow interp
  ASSERT_NEAR(pg.getError(), error, 1e-5);

  pg.addPrior(10, {prior});
  error = pg.getError();
  pg.update();
  ASSERT_TRUE(pg.getError() < error);
  ASSERT_TRUE(pg.getError() > 1);
}

TEST(pose_graph, test_lin_interp) {
  PoseGraph::Params pg_p;
  pg_p.num_frames_opt = 0;
  pg_p.allow_interpolation = true;
  PoseGraph pg(pg_p);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pg.addNode(0, pose);
  pose.translate(Eigen::Vector3d(1,0,0));
  pg.addNode(10, pose);
  Eigen::Isometry2d prior = Eigen::Isometry2d::Identity();
  prior.translate(Eigen::Vector2d(5,0));
  pg.addPrior(5, {prior});
  prior.translate(Eigen::Vector2d(2,0));
  pg.addPrior(15, {prior});

  pg.update();
  ASSERT_NEAR(pg.getPoseAtTime(0)->translation()[0], 5, 1e-5);

  // Reset pose graph
  pg = PoseGraph(pg_p);
  // Same drill, mix up the numbers and order
  prior = Eigen::Isometry2d::Identity();
  prior.translate(Eigen::Vector2d(0,1));
  pg.addPrior(4, {prior});
  prior.rotate(Eigen::Rotation2Dd(pi/2));
  pg.addPrior(7, {prior});
  pose = Eigen::Isometry3d::Identity();
  pg.addNode(0, pose);
  pg.addNode(5, pose);

  pg.update();
  ASSERT_NEAR(pg.getPoseAtTime(0)->translation()[1], 1, 1e-5);
  ASSERT_NEAR(Eigen::AngleAxisd(pg.getPoseAtTime(0)->rotation()).angle(), pi/6, 1e-5);
}

TEST(pose_graph, test_exact_interp) {
  PoseGraph::Params pg_p;
  pg_p.num_frames_opt = 0;
  pg_p.allow_interpolation = true;
  PoseGraph pg(pg_p);

  PoseGraph::Prior2D prior{};
  prior.pose.translate(Eigen::Vector2d(10,0));
  prior.pose.rotate(Eigen::Rotation2Dd(pi/2));

  pg.addNode(0, prior.local_pose);
  auto prior2 = prior;
  prior2.pose.translate(Eigen::Vector2d(1, 0));
  prior2.local_pose.translate(Eigen::Vector3d(1, 0, 0));
  pg.addPrior(1, prior2);
  pg.update();
  ASSERT_NEAR((pg.getPoseAtTime(0)->translation() - 
        Eigen::Vector3d(10, 0, 0)).norm(), 0, 1e-5);

  auto prior3 = prior2;
  prior3.pose.translate(Eigen::Vector2d(5, 0));
  prior3.local_pose.translate(Eigen::Vector3d(5, 0, 0));
  pg.addNode(2, prior3.local_pose);
  pg.update();

  ASSERT_NEAR(pg.getError(), 0, 1e-5);
  ASSERT_NEAR((pg.getPoseAtTime(0)->translation() - 
        Eigen::Vector3d(10, 0, 0)).norm(), 0, 1e-5);
  ASSERT_NEAR((pg.getPoseAtTime(2)->translation() - 
        Eigen::Vector3d(10, 6, 0)).norm(), 0, 1e-5);
}

} // namespace spomp
