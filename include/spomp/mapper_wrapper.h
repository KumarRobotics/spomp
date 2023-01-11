#pragma once

#include <ros/ros.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
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
    void panoCallback(const sensor_msgs::Image::ConstPtr& img_msg,
      const sensor_msgs::CameraInfo::ConstPtr& info_msg);
    void globalEstCallback(
        const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& est_msg);
    void odomCallback(
        const geometry_msgs::PoseStamped::ConstPtr &odom_msg);
    void semPanoCallback(const sensor_msgs::Image::ConstPtr& sem_img_msg);

    void visualize(const ros::TimerEvent& timer = {});
    void vizPoseGraph(const ros::Time& stamp);
    void publishOdomCorrection(const ros::Time& stamp);

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    tf2_ros::StaticTransformBroadcaster tf_static_broadcaster_{};

    // Timers
    ros::Timer viz_timer_;

    // Pubs
    ros::Publisher graph_viz_pub_;
    ros::Publisher map_pub_;

    // Subs
    image_transport::CameraSubscriber pano_sub_;
    ros::Subscriber est_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber sem_pano_sub_;

    // Objects
    Mapper mapper_;
    std::list<geometry_msgs::PoseStamped::ConstPtr> odom_buf_;

    // Static because read in static functions
    static std::string odom_frame_;
    static std::string map_frame_;
    static int viz_thread_period_ms_;

    // Timers
    Timer* viz_t_;
};

} // namespace spomp
