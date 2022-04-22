#pragma once

#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include "spomp/mapper.h"

namespace spomp {

class MapperWrapper {
  public:
    static Mapper createMapper(ros::NodeHandle& nh);
    MapperWrapper(ros::NodeHandle& nh);

    void initialize();

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_{};

    // Pubs
    ros::Publisher graph_viz_pub_;

    // Subs
    ros::Subscriber odom_sub_;
    ros::Subscriber global_est_sub_;

    // Objects
    Mapper mapper_;

    // Timers
    Timer* viz_t_;
};

} // namespace spomp
