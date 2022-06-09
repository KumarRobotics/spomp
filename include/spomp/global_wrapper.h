#pragma once

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include "spomp/global.h"

namespace spomp {

class GlobalWrapper {
  public:
    static Global createGlobal(ros::NodeHandle& nh);
    GlobalWrapper(ros::NodeHandle& nh);

    //! Startup subscribers
    void initialize();

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void mapSemImgCallback(const sensor_msgs::Image::ConstPtr& img_msg);
    void mapSemImgCenterCallback(const geometry_msgs::PointStamped::ConstPtr& pt_msg);
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal_msg);

    void processMapBuffers();
    void printTimings();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;

    // Pubs
    ros::Publisher local_goal_pub_;
    ros::Publisher graph_viz_pub_;
    ros::Publisher path_viz_pub_;

    // Subs
    ros::Subscriber map_sem_img_sub_;
    ros::Subscriber map_sem_img_center_sub_;
    ros::Subscriber goal_sub_;

    // Objects
    Global global_;

    long last_map_stamp_{0};
    std::map<long, const sensor_msgs::Image::ConstPtr> map_sem_buf_{};
    std::map<long, const geometry_msgs::PointStamped::ConstPtr> map_loc_buf_{};
};

} // namespace spomp
