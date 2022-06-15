#pragma once

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <actionlib/server/simple_action_server.h>
#include <spomp/GlobalNavigateAction.h>
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
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg);
    void goalSimpleCallback(const geometry_msgs::PoseStamped::ConstPtr& goal_msg);
    bool setGoal(const geometry_msgs::PoseStamped& goal_msg);
    void globalNavigateGoalCallback();
    void globalNavigatePreemptCallback();

    void processMapBuffers();
    void publishLocalGoal(const ros::Time& stamp);
    void visualizeGraph(const ros::Time& stamp);
    void visualizePath(const ros::Time& stamp);
    void printTimings();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Pubs
    ros::Publisher local_goal_pub_;
    ros::Publisher graph_viz_pub_;
    ros::Publisher path_viz_pub_;
    ros::Publisher map_img_viz_pub_;

    // Subs
    ros::Subscriber map_sem_img_sub_;
    ros::Subscriber map_sem_img_center_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber goal_sub_;
    
    // Action server
    actionlib::SimpleActionServer<spomp::GlobalNavigateAction> global_navigate_as_;

    // Objects
    Global global_;

    uint64_t last_map_stamp_{0};
    std::map<uint64_t, const sensor_msgs::Image::ConstPtr> map_sem_buf_{};
    std::map<uint64_t, const geometry_msgs::PointStamped::ConstPtr> map_loc_buf_{};

    Eigen::Vector2f last_goal_{0, 0};
    bool using_action_server_{false};

    // Config related
    // Static because read in static functions
    static std::string odom_frame_;
    static std::string map_frame_;
};

} // namespace spomp
