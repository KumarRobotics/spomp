#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace spomp {

struct Keyframe {
  long stamp = 0;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  cv::Mat depth_pano;
  cv::Mat intensity_pano;
};
  
} // namespace spomp
