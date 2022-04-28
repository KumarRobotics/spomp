#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include "spomp/utils.h"

namespace spomp {

template<typename T>
auto ROS2Eigen(const geometry_msgs::Pose& pose_msg) {
  using Iso = Eigen::Transform<T, 3, Eigen::Isometry>;
  Iso pose = Iso::Identity();
  pose.translate(Eigen::Matrix<T, 3, 1>(
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z
        ));
  pose.rotate(Eigen::Quaternion<T>(
        pose_msg.orientation.w,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z
        ));
  return pose;
}

template<typename T>
auto ROS2Eigen(const geometry_msgs::PoseStamped& pose_msg) {
  return ROS2Eigen<T>(pose_msg.pose);
}

template<typename T>
auto ROS2Eigen(const geometry_msgs::Transform& trans_msg) {
  using Iso = Eigen::Transform<T, 3, Eigen::Isometry>;
  Iso pose = Iso::Identity();
  pose.translate(Eigen::Matrix<T, 3, 1>(
        trans_msg.translation.x,
        trans_msg.translation.y,
        trans_msg.translation.z
        ));
  pose.rotate(Eigen::Quaternion<T>(
        trans_msg.rotation.w,
        trans_msg.rotation.x,
        trans_msg.rotation.y,
        trans_msg.rotation.z
        ));
  return pose;
}

template<typename T>
auto ROS2Eigen(const geometry_msgs::TransformStamped& pose_msg) {
  return ROS2Eigen<T>(pose_msg.transform);
}

template<typename T>
geometry_msgs::TransformStamped Eigen2ROS(
    const Eigen::Transform<T, 3, Eigen::Isometry>& pose) 
{
  geometry_msgs::TransformStamped pose_msg{};
  pose_msg.transform.translation.x = pose.translation()[0];
  pose_msg.transform.translation.y = pose.translation()[1];
  pose_msg.transform.translation.z = pose.translation()[2];
  Eigen::Quaternion<T> quat(pose.rotation());
  pose_msg.transform.rotation.x = quat.x();
  pose_msg.transform.rotation.y = quat.y();
  pose_msg.transform.rotation.z = quat.z();
  pose_msg.transform.rotation.w = quat.w();
  return pose_msg;
}

template<typename T>
geometry_msgs::PoseStamped Eigen2ROS(
    const Eigen::Transform<T, 2, Eigen::Isometry>& pose) 
{
  geometry_msgs::PoseStamped pose_msg{};
  pose_msg.pose.position.x = pose.translation()[0];
  pose_msg.pose.position.y = pose.translation()[1];
  pose_msg.pose.position.z = 0;
  Eigen::Rotation2D<T> rot(pose.rotation());
  Eigen::Quaternion<T> quat(Eigen::AngleAxis<T>(rot.angle(), Eigen::Matrix<T, 3, 1>::UnitZ()));
  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();
  return pose_msg;
}

template<typename T>
geometry_msgs::Twist Eigen2ROS(const Twist<T>& twist) {
  geometry_msgs::Twist twist_msg{};
  twist_msg.linear.x = twist.linear();
  twist_msg.linear.y = 0;
  twist_msg.linear.z = 0;
  twist_msg.angular.x = 0;
  twist_msg.angular.y = 0;
  twist_msg.angular.z = twist.ang();
  return twist_msg;
}

template<typename T>
geometry_msgs::Point Eigen2ROS(const Eigen::Matrix<T, 3, 1>& pos) {
  geometry_msgs::Point point;
  point.x = pos[0];
  point.y = pos[1];
  point.z = pos[2];
  return point;
}

} // namespace spomp
