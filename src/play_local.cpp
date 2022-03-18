#include <ros/ros.h>
#include "spomp/local_wrapper.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "spomp_local");
  ros::NodeHandle nh("~");

  try {
    spomp::LocalWrapper node(nh);
    node.play();
  } catch (const std::exception& e) {
    ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
  }
  return 0;
}

