#include <iostream>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include "spomp/terrain_pano.h"
#include "spomp/utils.h"

namespace spomp {

// Wrapper to expose protected methods to test
class TerrainPanoTester : TerrainPano {
  public:
    TerrainPanoTester(const TerrainPano::Params& p) : TerrainPano(p) {}
    using TerrainPano::getPano;
    using TerrainPano::updatePano;
    using TerrainPano::fillHoles;
    using TerrainPano::computeCloud;
    using TerrainPano::computeGradient;
    using TerrainPano::threshold;
    using TerrainPano::inflate;
};

static void BM_mod(benchmark::State& state) {
  for (auto _ : state) {
    for (int a=0; a<100; a++) {
      for (int b=0; b<a*2; b++) {
        int res = b % a;
        benchmark::DoNotOptimize(res);
      }
    }
  }
}
BENCHMARK(BM_mod);

static void BM_mod_fast(benchmark::State& state) {
  for (auto _ : state) {
    for (int a=0; a<100; a++) {
      for (int b=0; b<a*2; b++) {
        int res = fast_mod(b, a);
        benchmark::DoNotOptimize(res);
      }
    }
  }
}
BENCHMARK(BM_mod_fast);

TEST(terrain_pano, test_fill_holes) {
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(3, 100);
  pano.row(0).setLinSpaced(100, 1, 100);
  Eigen::ArrayXXf pano_copy = pano;

  // Create hole
  pano.block<3, 5>(0, 10) = 0;
  ASSERT_TRUE((pano != pano_copy).any());

  TerrainPanoTester tp({});
  tp.fillHoles(pano);

  ASSERT_TRUE(((pano - pano_copy).abs() < 0.001).all());

  // Test wrap-around
  pano = 1;
  for (int i = 91; i<100; i++) {
    pano(0, i) = i - 90;
  }
  for (int i = 0; i<10; i++) {
    pano(0, i) = 10+i;
  }
  pano_copy = pano;
  pano.block<3, 8>(0, 92) = 0;
  pano.block<3, 9>(0, 0) = 0;
  ASSERT_TRUE((pano != pano_copy).any());
  tp.fillHoles(pano);
  ASSERT_TRUE(((pano - pano_copy).abs() < 0.001).all());
}

static void BM_terrain_pano_fill_holes(benchmark::State& state) {
  TerrainPanoTester tp({});
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(256, 1024);
  for (int row=0; row<pano.rows(); ++row) {
    pano.row(row).setLinSpaced(1024, 1, 1024);
  }
  Eigen::ArrayXXf pano_copy = pano;
  // Create hole
  for (int cnt=0; cnt<1000; cnt++) {
    pano.block<3, 10>(rand() % 256, rand() % 1000) = 0;
  }
  for (auto _ : state) {
    tp.fillHoles(pano);
  }
}
BENCHMARK(BM_terrain_pano_fill_holes);

TEST(terrain_pano, test_compute_cloud) {
  TerrainPano tp({});
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(101, 8);
  pano(0,0) = 10;
  tp.updatePano(pano, {});
  const auto& cloud = tp.getCloud();

  ASSERT_NEAR(cloud[0](50, 0), 1, 1e-10);
  ASSERT_NEAR(cloud[1](50, 0), 0, 1e-10);
  ASSERT_NEAR(cloud[2](50, 0), 0, 1e-10);

  ASSERT_NEAR(cloud[0](0, 0), 10*sqrt(2)/2, 1e-5);
  ASSERT_NEAR(cloud[0](0, 0), cloud[2](0, 0), 1e-5);
  ASSERT_NEAR(cloud[0](100, 0), -cloud[2](100, 0), 1e-5);

  ASSERT_NEAR(cloud[0](50, 2), 0, 1e-5);
  ASSERT_NEAR(cloud[1](50, 2), 1, 1e-5);
  ASSERT_NEAR(cloud[2](50, 2), 0, 1e-5);

  ASSERT_NEAR(cloud[0](50, 4), -1, 1e-5);
  ASSERT_NEAR(cloud[1](50, 4), 0, 1e-5);
  ASSERT_NEAR(cloud[2](50, 4), 0, 1e-5);
}

static void BM_terrain_pano_compute_cloud(benchmark::State& state) {
  TerrainPanoTester tp({});
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(256, 1024);
  for (int row=0; row<pano.rows(); ++row) {
    pano.row(row).setLinSpaced(1024, 1, 1024);
  }
  tp.updatePano(pano, {});
  for (auto _ : state) {
    tp.computeCloud();
  }
}
BENCHMARK(BM_terrain_pano_compute_cloud);

TEST(terrain_pano, test_grad_thresh) {
  cv::Mat pano = cv::imread(ros::package::getPath("spomp") + 
                           "/test/pano.png", cv::IMREAD_ANYDEPTH);
  Eigen::MatrixXf pano_eig;
  cv::cv2eigen(pano, pano_eig);
  pano_eig /= 512;

  TerrainPanoTester tp({});
  tp.updatePano(pano_eig, {});
  Eigen::MatrixXf grad = tp.computeGradient().matrix();
  Eigen::MatrixXi thresh = tp.threshold(grad.array()).matrix();

  // This is kind of cheating.  Just write the image to disk
  // and manually review for correctness
  cv::Mat grad_cv;
  cv::eigen2cv(grad, grad_cv);
  grad_cv *= 256;
  cv::imwrite("grad.png", grad_cv);

  cv::Mat thresh_cv;
  cv::eigen2cv(thresh, thresh_cv);
  thresh_cv *= 256;
  cv::imwrite("thresh.png", thresh_cv);

  Eigen::MatrixXf filled = tp.getPano().matrix();
  cv::Mat filled_cv;
  cv::eigen2cv(filled, filled_cv);
  cv::imwrite("filled.png", filled_cv);
}

static void BM_terrain_pano_compute_gradient(benchmark::State& state) {
  TerrainPanoTester tp({});
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(256, 1024);
  for (int row=0; row<pano.rows(); ++row) {
    pano.row(row).setLinSpaced(1024, 1, 1024);
  }
  tp.updatePano(pano, {});
  for (auto _ : state) {
    tp.computeGradient();
  }
}
BENCHMARK(BM_terrain_pano_compute_gradient);

static void BM_terrain_pano_thresh(benchmark::State& state) {
  TerrainPanoTester tp({});
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Zero(256, 1024);
  // Create noise
  for (int cnt=0; cnt<1000; cnt++) {
    pano.block<1, 2>(rand() % 256, rand() % 1000) = 100;
  }
  for (auto _ : state) {
    tp.threshold(pano);
  }
}
BENCHMARK(BM_terrain_pano_thresh);

} // namespace spomp
