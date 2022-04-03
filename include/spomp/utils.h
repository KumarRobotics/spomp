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

inline Eigen::Vector2f cart2polar(const Eigen::Vector2f& cart) {
  return {cart.norm(), atan2(cart[1], cart[0])};
}

inline Eigen::Vector2f polar2cart(const Eigen::Vector2f& pol) {
  return {pol[0] * cos(pol[1]), pol[0] * sin(pol[1])};
}

inline float regAngle(float angle) {
  if (angle < 0) {
    angle += 2 * pi;
  } else if (angle >= 2 * pi) {
    angle -= 2 * pi;
  }
  return angle;
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

struct AngularProj {
  float start_angle{};
  float delta_angle{};
  float num{};

  AngularProj(int n, float s, float d) : num(n), start_angle(s), delta_angle(d) {};
  AngularProj(float s, float f, int n) {
    start_angle = s;
    delta_angle = (f - s)/n;
    num = n;
  }
  AngularProj() = default;

  int indAt(float angle) const {
    return std::roundf((regAngle(angle) - start_angle)/delta_angle);
  }

  float angAt(int ind) const {
    return start_angle + delta_angle * ind;
  }

  Eigen::VectorXf getAngles() const {
    return Eigen::VectorXf::LinSpaced(num, 
        start_angle, start_angle + num*delta_angle);
  }
};
  
} // namespace spomp
