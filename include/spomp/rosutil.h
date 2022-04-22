#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include "spomp/utils.h"

namespace spomp {

Eigen::Isometry3f ROS2Eigen(const geometry_msgs::PoseStamped& pose_msg) {
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translate(Eigen::Vector3f(
        pose_msg.pose.position.x,
        pose_msg.pose.position.y,
        pose_msg.pose.position.z
        ));
  pose.rotate(Eigen::Quaternionf(
        pose_msg.pose.orientation.w,
        pose_msg.pose.orientation.x,
        pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z
        ));
  return pose;
}

Eigen::Isometry3f ROS2Eigen(const geometry_msgs::TransformStamped& trans_msg) {
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translate(Eigen::Vector3f(
        trans_msg.transform.translation.x,
        trans_msg.transform.translation.y,
        trans_msg.transform.translation.z
        ));
  pose.rotate(Eigen::Quaternionf(
        trans_msg.transform.rotation.w,
        trans_msg.transform.rotation.x,
        trans_msg.transform.rotation.y,
        trans_msg.transform.rotation.z
        ));
  return pose;
}

geometry_msgs::TransformStamped Eigen2ROS(const Eigen::Isometry3f& pose) {
  geometry_msgs::TransformStamped pose_msg{};
  pose_msg.transform.translation.x = pose.translation()[0];
  pose_msg.transform.translation.y = pose.translation()[1];
  pose_msg.transform.translation.z = pose.translation()[2];
  Eigen::Quaternionf quat(pose.rotation());
  pose_msg.transform.rotation.x = quat.x();
  pose_msg.transform.rotation.y = quat.y();
  pose_msg.transform.rotation.z = quat.z();
  pose_msg.transform.rotation.w = quat.w();
  return pose_msg;
}

geometry_msgs::PoseStamped Eigen2ROS(const Eigen::Isometry2f& pose) {
  geometry_msgs::PoseStamped pose_msg{};
  pose_msg.pose.position.x = pose.translation()[0];
  pose_msg.pose.position.y = pose.translation()[1];
  pose_msg.pose.position.z = 0;
  Eigen::Rotation2Df rot(pose.rotation());
  Eigen::Quaternionf quat(Eigen::AngleAxisf(rot.angle(), Eigen::Vector3f::UnitZ()));
  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();
  return pose_msg;
}

geometry_msgs::Twist Eigen2ROS(const Twistf& twist) {
  geometry_msgs::Twist twist_msg{};
  twist_msg.linear.x = twist.linear();
  twist_msg.linear.y = 0;
  twist_msg.linear.z = 0;
  twist_msg.angular.x = 0;
  twist_msg.angular.y = 0;
  twist_msg.angular.z = twist.ang();
  return twist_msg;
}

} // namespace spomp
