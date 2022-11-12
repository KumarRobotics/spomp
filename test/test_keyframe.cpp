#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include "spomp/keyframe.h"
#include "spomp/utils.h"

namespace spomp {

TEST(keyframe, test_get_point_cloud) {
  cv::Mat depth(100, 1000, CV_32F, cv::Scalar(0));
  cv::Mat intensity(100, 1000, CV_16U, cv::Scalar(0));
  Keyframe::setIntrinsics(deg2rad(90), depth.size());

  depth.at<float>(0, 0) = 10;
  depth.at<float>(0, 250) = 20;
  depth.at<float>(99, 0) = 30;
  depth.at<float>(99, 500) = 40;

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translate(Eigen::Vector3d{1, 0, 0});
  Keyframe k{0, pose, depth, intensity};

  auto cloud = k.getPointCloud();
  ASSERT_EQ(cloud.cols(), 4);
  ASSERT_EQ(cloud.rows(), 5);

  EXPECT_NEAR(cloud.col(0)[0], 10*cos(deg2rad(45)) + 1, 1e-5);
  EXPECT_NEAR(cloud.col(0)[1], 0, 1e-5);
  EXPECT_NEAR(cloud.col(0)[2], 10*sin(deg2rad(45)), 1e-5);

  EXPECT_NEAR(cloud.col(1)[0], 1, 1e-5);
  EXPECT_NEAR(cloud.col(1)[1], -20*cos(deg2rad(45)), 1e-5);
  EXPECT_NEAR(cloud.col(1)[2], 20*sin(deg2rad(45)), 1e-5);

  EXPECT_NEAR(cloud.col(2)[0], 30*cos(deg2rad(45)) + 1, 1e-5);
  EXPECT_NEAR(cloud.col(2)[1], 0, 1e-5);
  EXPECT_NEAR(cloud.col(2)[2], -30*sin(deg2rad(45)), 1e-5);

  EXPECT_NEAR(cloud.col(3)[0], -40*cos(deg2rad(45)) + 1, 1e-5);
  EXPECT_NEAR(cloud.col(3)[1], 0, 1e-5);
  EXPECT_NEAR(cloud.col(3)[2], -40*sin(deg2rad(45)), 1e-5);
}

static void BM_keyframe_get_point_cloud(benchmark::State& state) {
  cv::Mat pano = cv::imread(ros::package::getPath("spomp") + 
                           "/test/pano.png", cv::IMREAD_ANYDEPTH);
  cv::Mat rescaled_depth;
  pano.convertTo(rescaled_depth, CV_32F, 1./512);

  Keyframe::setIntrinsics(deg2rad(90), rescaled_depth.size());
  Keyframe k{0, Eigen::Isometry3d::Identity(), rescaled_depth, cv::Mat()};
  for (auto _ : state) {
    auto cloud = k.getPointCloud();
    benchmark::DoNotOptimize(cloud);
  }
}
BENCHMARK(BM_keyframe_get_point_cloud);

} // namespace spomp
