#include <ros/ros.h>
#include "spomp/global_wrapper.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "spomp_global");
  ros::NodeHandle nh("~");

  try {
    spomp::GlobalWrapper node(nh);
    node.initialize();
  } catch (const std::exception& e) {
    ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
  }
  return 0;
}

