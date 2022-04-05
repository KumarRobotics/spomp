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

  AngularProj(int n, float s, float d) : num(n), start_angle(s), delta_angle(d) {};
  AngularProj(float s, float f, int n) {
    start_angle = s;
    // (n-1) spaces for n points
    delta_angle = (f - s)/(n-1);
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
