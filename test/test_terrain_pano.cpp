#include <iostream>
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include "spomp/terrain_pano.h"
#include "spomp/utils.h"

namespace spomp {

// Wrapper to expose protected methods to test
class TerrainPanoTester : TerrainPano {
  public:
    TerrainPanoTester(const TerrainPano::Params& p) : TerrainPano(p) {}
    using TerrainPano::fillHoles;
    using TerrainPano::computeCloud;
    using TerrainPano::computeGradient;
    using TerrainPano::threshold;
    using TerrainPano::inflate;
};

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

static void BM_terrain_pano_fill_holes(benchmark::State& state) {
  TerrainPanoTester tp({});
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(128, 1024);
  for (int row=0; row<pano.rows(); ++row) {
    pano.row(row).setLinSpaced(1024, 1, 1024);
  }
  Eigen::ArrayXXf pano_copy = pano;
  // Create hole
  for (int cnt=0; cnt<1000; cnt++) {
    pano.block<3, 10>(rand() % 128, rand() % 1000) = 0;
  }
  for (auto _ : state) {
    tp.fillHoles(pano);
  }
}
BENCHMARK(BM_terrain_pano_fill_holes);

} // namespace spomp
