#include "spomp/keyframe.h"

namespace spomp {

Keyframe::Keyframe(long stamp, const Eigen::Isometry3d& pose, 
    const cv::Mat& d_p, const cv::Mat& i_p) :
  stamp_(stamp),
  pose_(pose),
  depth_pano_(d_p),
  intensity_pano_(i_p) {}

} // namespace spomp
