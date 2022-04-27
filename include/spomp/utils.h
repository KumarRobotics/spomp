#pragma once

//! Various helper functions, mostly math primitives

#include <Eigen/Dense>

namespace spomp {

constexpr double pi = 3.14159265358979323846;

inline int fast_mod(int a, int b) noexcept {
  // This is 3-6 times faster than %
  // Only valid for a < 2*b
  return a < b ? a : a - b;
}

inline float deg2rad(float d) noexcept {
  return d * pi / 180;
}

inline float rad2deg(float r) noexcept {
  return r * 180 / pi;
}

inline Eigen::Vector2f cart2polar(const Eigen::Vector2f& cart) {
  return {cart.norm(), atan2(cart[1], cart[0])};
}

inline Eigen::Vector2f polar2cart(const Eigen::Vector2f& pol) {
  return {pol[0] * cos(pol[1]), pol[0] * sin(pol[1])};
}

inline Eigen::Isometry3d pose22pose3(const Eigen::Isometry2d& pose_2) {
  Eigen::Isometry3d pose_3 = Eigen::Isometry3d::Identity();
  pose_3.translation().head<2>() = pose_2.translation();
  pose_3.rotate(Eigen::AngleAxisd(
        Eigen::Rotation2Dd(pose_2.rotation()).angle(), 
                           Eigen::Vector3d::UnitZ()));
  return pose_3;
}

inline Eigen::Isometry2d pose32pose2(const Eigen::Isometry3d& pose_3) {
  Eigen::Isometry2d pose_2 = Eigen::Isometry2d::Identity();
  pose_2.translation() = pose_3.translation().head<2>();
  Eigen::Vector3d rot_x = pose_3.rotation() * Eigen::Vector3d::UnitX();
  pose_2.rotate(Eigen::Rotation2D(atan2(rot_x[1], rot_x[0])));
  return pose_2;
}

inline float regAngle(float angle) {
  if (angle < 0) {
    angle += 2 * pi;
  } else if (angle >= 2 * pi) {
    angle -= 2 * pi;
  }
  return angle;
}

inline float crossNorm(const Eigen::Vector2f& v1, const Eigen::Vector2f &v2) {
  Eigen::Vector3f v1_3 = Eigen::Vector3f::Zero();
  Eigen::Vector3f v2_3 = Eigen::Vector3f::Zero();
  v1_3.head<2>() = v1;
  v2_3.head<2>() = v2;
  return v1_3.cross(v2_3).norm();
}

//! Simple wrapper class to abstract Twist
template <typename T>
class Twist {
  using Vector2T = Eigen::Matrix<T, 2, 1>;
  public:
    Twist(const Vector2T& twist) : twist_(twist) {}
    Twist(T lin, T ang) : twist_{lin, ang} {}
    Twist() = default;

    T linear() const {
      return twist_[0];
    }

    T ang() const {
      return twist_[1];
    }

    T& linear() {
      return twist_[0];
    }

    T& ang() {
      return twist_[1];
    }

    Twist& operator+=(const Twist& rhs) {
      twist_ += rhs.twist_;
      return *this;
    }

    Twist& operator-=(const Twist& rhs) {
      twist_ -= rhs.twist_;
      return *this;
    }

    Twist operator-() const {
      return Twist(-twist_);
    }

    // Have to create new Twist anyway, so pass by value
    friend Twist operator+(Twist lhs, const Twist& rhs) {
      lhs += rhs;
      return lhs;
    }

    friend Twist operator-(Twist lhs, const Twist& rhs) {
      lhs -= rhs;
      return lhs;
    }
  private:
    Vector2T twist_{Vector2T::Zero()};
};

using Twistf = Twist<float>;

struct AngularProj {
  float start_angle{};
  float delta_angle{};
  int num{};

  struct StartDelta {
    float s{};
    float d{};
  };
  AngularProj(const StartDelta& sd, int n) : num(n), start_angle(sd.s), delta_angle(sd.d) {};
  struct StartFinish {
    float s{};
    float f{};
  };
  AngularProj(const StartFinish& sf, int n) {
    start_angle = sf.s;
    // (n-1) spaces for n points
    delta_angle = (sf.f - sf.s)/(n-1);
    num = n;
  }
  AngularProj() = default;

  int indAt(float angle) const {
    int ind = std::roundf((regAngle(angle) - start_angle)/delta_angle);
    while (ind >= num) {
      ind -= num; 
    }
    while (ind < 0) {
      ind += num; 
    }
    return ind;
  }

  float angAt(int ind) const {
    return start_angle + delta_angle * ind;
  }

  Eigen::VectorXf getAngles() const {
    return Eigen::VectorXf::LinSpaced(num, 
        start_angle, start_angle + (num-1)*delta_angle);
  }
};
  
} // namespace spomp
