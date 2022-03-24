#include <benchmark/benchmark.h>

static void BM_test(benchmark::State& state) {
  for (auto _ : state) {
    int b = 0;
    for (int a=0; a<1000; a++) {
      b += a;
    }
  }
}

BENCHMARK(BM_test);

BENCHMARK_MAIN();
