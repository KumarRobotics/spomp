#include "spomp/keyframe.h"

namespace spomp {

Keyframe::Keyframe(long stamp, const Eigen::Isometry3d& pose, 
    const cv::Mat& d_p, const cv::Mat& i_p) :
  stamp_(stamp),
  pose_(pose),
  depth_pano_(d_p),
  intensity_pano_(i_p) {}

PointCloudArray Keyframe::getPointCloud() const {
  PointCloudArray cloud(5, depth_pano_.total());
  return cloud;
}

} // namespace spomp
