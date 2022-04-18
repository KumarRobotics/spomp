#include "spomp/mapper_wrapper.h"

namespace spomp {

MapperWrapper::MapperWrapper(ros::NodeHandle& nh) : nh_(nh) {}

void MapperWrapper::initialize() {
  ros::spin();
}

} // namespace spomp
