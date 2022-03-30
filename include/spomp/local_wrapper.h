#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "spomp/local.h"
#include "spomp/remote.h"

namespace spomp {

class LocalWrapper {
  public:
    static Local createLocal(ros::NodeHandle& nh);
    LocalWrapper(ros::NodeHandle& nh);

    //! Manually run through dataset
    void play();

    //! Startup subscribers
    void initialize();

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void panoCallback(const sensor_msgs::Image::ConstPtr& img_msg);
    void visualizePano(const ros::Time& stamp);
    void visualizeCloud(const ros::Time& stamp);
    void visualizeReachability(const ros::Time& stamp);

    void printTimings();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
    ros::Publisher obs_pano_viz_pub_;
    ros::Publisher obs_cloud_viz_pub_;
    ros::Publisher reachability_viz_pub_;

    ros::Subscriber pano_sub_;

    Local local_;

    Remote remote_;

    std::string pano_frame_{"pano"};

    // Timers
    Timer* viz_pano_t_{};
    Timer* viz_cloud_t_{};
};

}
