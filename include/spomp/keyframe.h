#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace spomp {

class Keyframe {
  public:
    Keyframe(long stamp, const Eigen::Isometry3d& pose, 
        const cv::Mat& d_p, const cv::Mat& i_p);

    auto getStamp() const {
      return stamp_;
    }

    const auto& getPose() const {
      return pose_;
    }

    void setPose(const Eigen::Isometry3d& p) {
      pose_ = p;
    }

  private:
    long stamp_{0};
    Eigen::Isometry3d pose_{Eigen::Isometry3d::Identity()};
    cv::Mat depth_pano_;
    cv::Mat intensity_pano_;
};
  
} // namespace spomp
