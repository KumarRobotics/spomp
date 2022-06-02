#include "spomp/global_wrapper.h"

namespace spomp {

GlobalWrapper::GlobalWrapper(ros::NodeHandle& nh) : nh_(nh) {
}

Global GlobalWrapper::createGlobal(ros::NodeHandle& nh) {
  // Load ROS params here
  return Global();
}

void GlobalWrapper::initialize() {
  ros::spin();
}

} // namespace spomp
