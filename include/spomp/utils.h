#pragma once

//! Various helper functions, mostly math primitives

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <iostream>

namespace spomp {

constexpr double pi = 3.14159265358979323846;

/**
 * @brief Calculates the remainder of the division between two integers.
 *
 * This function calculates the remainder of the division between the dividend 'a' and the divisor 'b'.
 * It is an optimized implementation that avoids the division operator to achieve faster performance.
 *
 * @param a The dividend.
 * @param b The divisor.
 *
 * @return The remainder of the division between 'a' and 'b'.
 *
 * @note This function assumes that 'b' is not zero. If 'b' is zero, the behavior is undefined.
 */
    inline int fast_mod(int a, int b) noexcept {
  // This is 3-6 times faster than %
  // Only valid for a < 2*b
  return a < b ? a : a - b;
}

/**
 * @brief Converts degrees to radians.
 *
 * This function takes a degree value and converts it to radians.
 *
 * @param d The angle in degrees.
 * @return The corresponding angle in radians.
 */
inline float deg2rad(float d) noexcept {
  return d * pi / 180;
}

/**
 * @brief Converts radians to degrees.
 *
 * This function takes a float value representing an angle in radians and converts it to degrees.
 * The result is returned as a float value.
 *
 * @param r The angle in radians.
 * @return The angle in degrees.
 */
inline float rad2deg(float r) noexcept {
  return r * 180 / pi;
}

/**
 * @brief Converts Cartesian coordinates to polar coordinates.
 *
 * @param cart The Cartesian coordinates in the form of a 2D vector.
 * @return Eigen::Vector2f The polar coordinates in the form of a 2D vector.
 */
inline Eigen::Vector2f cart2polar(const Eigen::Vector2f& cart) {
  return {cart.norm(), atan2f(cart[1], cart[0])};
}

/**
 * @brief Convert polar coordinates to Cartesian coordinates
 *
 * This function takes a 2D vector representing polar coordinates and returns a 2D vector
 * representing Cartesian coordinates.
 *
 * @param pol A 2D vector representing polar coordinates, where pol[0] is the radius and
 *            pol[1] is the angle in radians.
 * @return A 2D vector representing Cartesian coordinates, where the x-component is pol[0]
 *         multiplied by the cosine of pol[1] and the y-component is pol[0] multiplied by the
 *         sine of pol[1].
 */
inline Eigen::Vector2f polar2cart(const Eigen::Vector2f& pol) {
  return {pol[0] * cosf(pol[1]), pol[0] * sinf(pol[1])};
}

/**
 * @brief Converts an HSV color value to RGB color value.
 *
 * This function takes an HSV color value as input and converts it to an RGB color value. The input color values should be in the range [0, 1].
 *
 * @param hsv The HSV color value to be converted.
 * @return The RGB color value corresponding to the input HSV color.
 */
// All values in [0, 1]
inline Eigen::Vector3f hsv2rgb(const Eigen::Vector3f& hsv) {
  // https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html
  Eigen::Vector3f rgb;

  float M = hsv[2];
  float m = M * (1 - hsv[1]);
  float z = (M - m) * (1 - std::abs(fmod(hsv[0] * 6, 2) - 1));

  if (hsv[0] < 1./6) {
    rgb[0] = M;
    rgb[1] = z + m;
    rgb[2] = m;
  } else if (hsv[0] < 2./6) {
    rgb[0] = z + m;
    rgb[1] = M;
    rgb[2] = m;
  } else if (hsv[0] < 3./6) {
    rgb[0] = m;
    rgb[1] = M;
    rgb[2] = z + m;
  } else if (hsv[0] < 4./6) {
    rgb[0] = m;
    rgb[1] = z + m;
    rgb[2] = M;
  } else if (hsv[0] < 5./6) {
    rgb[0] = z + m;
    rgb[1] = m;
    rgb[2] = M;
  } else {
    rgb[0] = M;
    rgb[1] = m;
    rgb[2] = z + m;
  }

  return rgb;
}

/**
 * @brief Converts a 2D pose to a 3D pose.
 *
 * Given a 2D pose, this function converts it to a 3D pose by assuming that the input pose has zero rotation
 * around the Z-axis. The resulting pose will have the same translation as the input pose and zero rotation around
 * the Y and X axes.
 *
 * @tparam T The scalar type.
 * @param pose_2 The input 2D pose.
 * @return The resulting 3D pose.
 */
template <typename T>
inline auto pose22pose3(const Eigen::Transform<T, 2, Eigen::Isometry>& pose_2) {
  using Isometry3T = Eigen::Transform<T, 3, Eigen::Isometry>;
  using Vector3T = Eigen::Matrix<T, 3, 1>;

  Isometry3T pose_3 = Isometry3T::Identity();
  pose_3.translation().head(2) = pose_2.translation();
  pose_3.rotate(Eigen::AngleAxis<T>(
        Eigen::Rotation2D<T>(pose_2.rotation()).angle(), 
                             Vector3T::UnitZ()));
  return pose_3;
}

/**
 * \brief Converts a 3D pose to a 2D pose by removing the z-axis information
 *
 * This function converts a 3D pose represented by Eigen::Transform to a 2D pose
 * represented by Eigen::Transform. The z-axis information is removed from the
 * translation component of the pose, and the rotation component is adjusted
 * accordingly.
 *
 * \tparam T The data type of the pose elements (float, double, etc.)
 * \param pose_3 The 3D pose to convert
 * \return The converted 2D pose
 */
template <typename T>
inline auto pose32pose2(const Eigen::Transform<T, 3, Eigen::Isometry>& pose_3) {
  using Isometry2T = Eigen::Transform<T, 2, Eigen::Isometry>;
  using Vector3T = Eigen::Matrix<T, 3, 1>;

  Isometry2T pose_2 = Isometry2T::Identity();
  // For some reason the templated version of head causes problems
  pose_2.translation() = pose_3.translation().head(2);
  Vector3T rot_x = pose_3.rotation() * Vector3T::UnitX();
  pose_2.rotate(Eigen::Rotation2D<T>(atan2(rot_x[1], rot_x[0])));
  return pose_2;
}

/**
 * @brief Regularizes an angle between 0 and 2π.
 *
 * This function takes an angle as input and normalizes it to be within the range of 0 to 2π.
 *
 * @param angle The input angle in radians.
 * @return The normalized angle between 0 and 2π.
 */
inline float regAngle(float angle) {
  if (angle < 0) {
    angle += 2 * pi;
  } else if (angle >= 2 * pi) {
    angle -= 2 * pi;
  }
  return angle;
}

/**
 * @brief Calculates the norm of the cross product of two 2D vectors.
 *
 * This function calculates the norm of the cross product of two 2D vectors (vectors in the XY-plane).
 *
 * @param v1 The first input vector.
 * @param v2 The second input vector.
 * @return The norm of the cross product of v1 and v2.
 */
inline float crossNorm(const Eigen::Vector2f& v1, const Eigen::Vector2f &v2) {
  Eigen::Vector3f v1_3 = Eigen::Vector3f::Zero();
  Eigen::Vector3f v2_3 = Eigen::Vector3f::Zero();
  v1_3.head<2>() = v1;
  v2_3.head<2>() = v2;
  return v1_3.cross(v2_3).norm();
}

/**
 * @class Twist
 * @brief Simple wrapper class to abstract Twist
 *
 * The Twist class is a simple wrapper around a 2D vector representing
 * a linear and angular motion. It provides convenient methods for accessing
 * and modifying the linear and angular components of the twist vector.
 */
//! Simple wrapper class to abstract Twist
template <typename T>
class Twist {
  using Vector2T = Eigen::Matrix<T, 2, 1>;
  public:
    explicit Twist(const Vector2T& twist) : twist_(twist) {}
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
using Twistd = Twist<double>;

/**
 * @struct AngularProj
 * @brief Represents an angular projection with a start angle, delta angle, and number of points
 */
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

/**
 * @brief The MapReferenceFrame struct represents a reference frame for a map.
 *
 * The MapReferenceFrame struct provides methods for converting coordinates between
 * the world coordinate system and the image coordinate system based on the map's
 * resolution, center, and size. It also provides a method for checking if an image
 * point is within the map boundaries and a method for computing the intersection
 * region between two map reference frames.
 */
struct MapReferenceFrame {
  float res{1};
  Eigen::Vector2f center{0, 0};
  // width (cols) x height (rows)
  Eigen::Vector2f size{0, 0};

  void setMapSizeFrom(const cv::Mat& map) {
    size = Eigen::Vector2f(map.cols, map.rows);
  }

  Eigen::Vector2f world2img(const Eigen::Vector2f& world_c) const {
    Eigen::Vector2f world_pt = world_c - center;
    Eigen::Vector2f img_pt = {-world_pt[1], -world_pt[0]};
    img_pt *= res;
    img_pt += size/2;
    return img_pt.array().round();
  }

  Eigen::Vector2f img2world(const Eigen::Vector2f& img_c) const {
    Eigen::Vector2f img_pt = img_c - size/2;
    img_pt /= res;
    Eigen::Vector2f world_pt = {-img_pt[1], -img_pt[0]};
    world_pt += center;
    return world_pt;
  }
  
  bool imgPointInMap(const Eigen::Vector2f& img_c) const {
    return ((img_c.array() >= 0).all() && 
            (img_c.array() < size.array()).all());
  }

  struct Intersect {
    cv::Rect old_frame;
    cv::Rect new_frame;
  };
  Intersect computeIntersect(const MapReferenceFrame& new_mrf) const {
    // Location of upper left corner of old map in new map
    Eigen::Vector2f ul_old_in_new = new_mrf.world2img(
        img2world({0, 0}));
    cv::Rect old_roi(cv::Point(ul_old_in_new[0], ul_old_in_new[1]), 
        cv::Size(size[0], size[1]));
    cv::Rect new_roi({}, cv::Size(new_mrf.size[0], new_mrf.size[1]));

    auto intersect = new_roi & old_roi;
    auto intersect_old_frame = intersect;
    intersect_old_frame -= old_roi.tl();

    return {intersect_old_frame, intersect};
  }
};
  
} // namespace spomp
