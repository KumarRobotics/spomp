#pragma once

#include <Eigen/Dense>

namespace spomp {

constexpr float pi = 3.14159265358979323846;

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
  private:
    Vector2T twist_{Vector2T::Zero()};
};

using Twistf = Twist<float>;
  
} // namespace spomp
