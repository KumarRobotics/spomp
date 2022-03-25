#include <benchmark/benchmark.h>
#include <string>

#include "spomp/utils.h"
#include "spomp/terrain_pano.h"

namespace spomp {

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
  TerrainPano tp({});
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

BENCHMARK_MAIN();
