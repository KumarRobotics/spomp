#include <benchmark/benchmark.h>
#include <string>

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

static void BM_mod_fast(benchmark::State& state) {
  for (auto _ : state) {
    for (int a=0; a<100; a++) {
      for (int b=0; b<a*2; b++) {
        int res = b <= a ? b : b - a;
        benchmark::DoNotOptimize(res);
      }
    }
  }
}

BENCHMARK(BM_mod);
BENCHMARK(BM_mod_fast);

BENCHMARK_MAIN();
