#pragma once

#include <memory>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace Eigen {
  using Array5Xf = Array<float, 5, Dynamic>;
  using Array6Xf = Array<float, 6, Dynamic>;
}

namespace spomp {

using PointCloudArray = Eigen::Array5Xf;

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

    auto getSize() const {
      return depth_pano_.size();
    }

    //! Return array of points and other point data
    PointCloudArray getPointCloud() const;

    void setPose(const Eigen::Isometry3d& p) {
      pose_ = p;
    }

    static void setIntrinsics(float vfov, const cv::Size& size);

  private:
    long stamp_{0};
    Eigen::Isometry3d pose_{Eigen::Isometry3d::Identity()};
    cv::Mat depth_pano_{};
    cv::Mat intensity_pano_{};
    Eigen::Isometry3d map_pose_{Eigen::Isometry3d::Identity()};

    static Eigen::Array3Xf projection_;
};
  
} // namespace spomp
