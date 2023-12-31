#include "spomp/keyframe.h"
#include "spomp/utils.h"

#include <iostream>

namespace spomp {

// Static members
Eigen::Array3Xf Keyframe::projection_;

Keyframe::Keyframe(uint64_t stamp, const Eigen::Isometry3d& pose, 
    const cv::Mat& d_p, const cv::Mat& i_p, const cv::Mat& s_p) :
  stamp_(stamp),
  pose_(pose),
  depth_pano_(d_p),
  intensity_pano_(i_p),
  sem_pano_(s_p) {};

PointCloudArray Keyframe::getPointCloud() const {
  PointCloudArray cloud = PointCloudArray::Zero(5, depth_pano_.total());
  cloud.row(4).setConstant(255);

  if (projection_.size() < 1) {
    return cloud;
  }

  int sparse_ind = 0;
  int dense_ind = 0;
  const float* range_ptr = nullptr;
  const uint16_t* intensity_ptr = nullptr;
  const uint8_t* sem_ptr = nullptr;
  for (int y=0; y<depth_pano_.rows; ++y) {
    range_ptr = depth_pano_.ptr<float>(y);
    if (intensity_pano_.size() == depth_pano_.size()) {
      intensity_ptr = intensity_pano_.ptr<uint16_t>(y);
    }
    if (sem_pano_.size() == depth_pano_.size()) {
      sem_ptr = sem_pano_.ptr<uint8_t>(y);
    }

    for (int x=0; x<depth_pano_.cols; ++x) {
      float range = range_ptr[x];
      if (range < 0.1) { 
        ++dense_ind;
        continue;
      }

      cloud.col(sparse_ind).head<3>() = pose_.cast<float>() * 
        (projection_.col(dense_ind) * range);
      if (intensity_ptr) {
        cloud.col(sparse_ind)[3] = intensity_ptr[x];
      }
      if (sem_ptr) {
        cloud.col(sparse_ind)[4] = sem_ptr[x];
      }
      ++dense_ind;
      ++sparse_ind;
    }
  }

  cloud.conservativeResize(Eigen::NoChange_t(), sparse_ind);
  return cloud;
}

void Keyframe::setIntrinsics(float fov, const cv::Size& size) {
  if (projection_.size() > 0) {
    return;
  }

  projection_ = Eigen::Array3Xf::Zero(3, size.area());

  AngularProj alt_p(AngularProj::StartFinish{
      fov/2, -fov/2}, size.height);
  // Go negative because the panorama wraps around CW
  // Then add 360 degrees to keep positive
  AngularProj az_p(AngularProj::StartFinish{
      2*pi, static_cast<float>(2*pi*(1./size.width))}, size.width);

  int index = 0;
  for (int y=0; y<size.height; ++y) {
    for (int x=0; x<size.width; ++x) {
      float alt = alt_p.angAt(y);
      float az = az_p.angAt(x);
      projection_.col(index) = Eigen::Vector3f{
          static_cast<float>(cos(alt) * cos(az)),
          static_cast<float>(cos(alt) * sin(az)),
          static_cast<float>(sin(alt))
        };
      ++index;
    }
  }
}

} // namespace spomp
