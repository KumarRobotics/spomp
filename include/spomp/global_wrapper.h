#pragma once

#include <ros/ros.h>
#include "spomp/global.h"

namespace spomp {

class GlobalWrapper {
  public:
    static Global createGlobal(ros::NodeHandle& nh);
    GlobalWrapper(ros::NodeHandle& nh);

    //! Startup subscribers
    void initialize();

  private:
    ros::NodeHandle nh_;
};

} // namespace spomp
