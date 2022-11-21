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
    Keyframe(uint64_t stamp, const Eigen::Isometry3d& pose, 
        const cv::Mat& d_p, const cv::Mat& i_p, const cv::Mat& s_p);

    auto getStamp() const {
      return stamp_;
    }

    const auto& getPose() const {
      return pose_;
    }

    const auto& getMapPose() const {
      return map_pose_;
    }

    auto getSize() const {
      return depth_pano_.size();
    }

    bool isOptimized() const {
      return optimized_;
    }

    bool inMap() const {
      return !map_pose_.matrix().isIdentity(1e-5);
    }

    //! Return array of points and other point data
    PointCloudArray getPointCloud() const;

    void setPose(const Eigen::Isometry3d& p) {
      pose_ = p;
    }

    void setOptimized() {
      optimized_ = true;
    }

    void setSem(const cv::Mat& sem) {
      sem_pano_ = sem;
    }

    void updateMapPose() {
      map_pose_ = pose_;
    }

    static void setIntrinsics(float vfov, const cv::Size& size);

  private:
    uint64_t stamp_{0};
    Eigen::Isometry3d pose_{Eigen::Isometry3d::Identity()};
    cv::Mat depth_pano_{};
    cv::Mat intensity_pano_{};
    cv::Mat sem_pano_{};
    Eigen::Isometry3d map_pose_{Eigen::Isometry3d::Identity()};
    bool optimized_{false};

    static Eigen::Array3Xf projection_;
};
  
} // namespace spomp
