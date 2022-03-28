#pragma once

#include <ros/ros.h>
#include "spomp/local.h"

namespace spomp {

class LocalWrapper {
  public:
    LocalWrapper(ros::NodeHandle& nh);

    //! Manually run through dataset
    void play();

    //! Startup subscribers
    void initialize();

  private:
    ros::NodeHandle nh_;
};

}
