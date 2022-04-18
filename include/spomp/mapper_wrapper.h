#pragma once

#include <ros/ros.h>

namespace spomp {

class MapperWrapper {
  public:
    MapperWrapper(ros::NodeHandle& nh);

    void initialize();

  private:
    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
};

} // namespace spomp
