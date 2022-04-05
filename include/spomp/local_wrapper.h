#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>
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

    static Eigen::Isometry3f ROS2Eigen(const geometry_msgs::TransformStamped& trans_msg);
    static geometry_msgs::TransformStamped Eigen2ROS(const Eigen::Isometry3f& pose);

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    static bool getControlTrans(Eigen::Isometry3f& trans);

    void panoCallback(const sensor_msgs::Image::ConstPtr& img_msg,
        const sensor_msgs::CameraInfo::ConstPtr& info_msg);
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg);

    void publishTransform(const ros::Time& stamp);
    void visualizePano(const ros::Time& stamp);
    void visualizeCloud(const ros::Time& stamp);
    void visualizeReachability(const ros::Time& stamp);
    void visualizeControl(const ros::Time& stamp);

    void printTimings();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // Pubs
    ros::Publisher obs_pano_viz_pub_;
    ros::Publisher obs_cloud_viz_pub_;
    ros::Publisher reachability_viz_pub_;
    ros::Publisher control_viz_pub_;
    ros::Publisher local_goal_viz_pub_;
    ros::Publisher control_pub_;

    // Subs
    image_transport::CameraSubscriber pano_sub_;
    ros::Subscriber goal_sub_;

    // Object pointers
    Local local_;

    Remote remote_;

    // Config related
    // Static because modified in static functions
    static std::string pano_frame_;
    static std::string body_frame_;
    static std::string control_frame_;

    // Timers
    Timer* viz_pano_t_{};
    Timer* viz_cloud_t_{};
};

}
