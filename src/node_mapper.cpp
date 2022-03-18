#include <ros/ros.h>
#include "spomp/mapper_wrapper.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "spomp_mapper");
  ros::NodeHandle nh("~");

  try {
    spomp::MapperWrapper node(nh);
    node.initialize();
  } catch (const std::exception& e) {
    ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
  }
  return 0;
}

