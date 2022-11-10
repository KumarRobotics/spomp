#include <benchmark/benchmark.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "spomp/keyframe.h"
#include "spomp/utils.h"

namespace spomp {

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
