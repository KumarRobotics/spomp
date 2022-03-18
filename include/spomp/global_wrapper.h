#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/transform_listener.h>
#include <actionlib/server/simple_action_server.h>
#include <spomp/GlobalNavigateAction.h>
#include <spomp/LocalReachabilityArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_msgs/GridMap.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
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
    void aerialMapCallback(const grid_map_msgs::GridMap::ConstPtr& map_msg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg);
    void goalSimpleCallback(const geometry_msgs::PoseStamped::ConstPtr& goal_msg);
    void reachabilityCallback(const sensor_msgs::LaserScan::ConstPtr& reachability_msg);
    void otherReachabilityCallback(int robot_id, 
        const LocalReachabilityArray::ConstPtr& reachability_msg);
    bool setGoal(const geometry_msgs::PoseStamped& goal_msg);
    void timeoutTimerCallback(const ros::TimerEvent& event);
    void globalNavigateGoalCallback();
    void globalNavigatePreemptCallback();
    void globalNavigateSetFailed();

    void publishLocalGoal(const ros::Time& stamp);
    void cancelLocalPlanner();
    void publishReachabilityHistory(const ros::Time& stamp);
    void visualizeGraph(const ros::Time& stamp);
    void visualizePath(const ros::Time& stamp);
    void visualizeAerialMap(const ros::Time& stamp);
    void printTimings();

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Pubs
    ros::Publisher local_goal_pub_;
    ros::Publisher reachability_history_pub_;
    ros::Publisher graph_viz_pub_;
    ros::Publisher path_viz_pub_;
    image_transport::Publisher map_img_viz_pub_;
    image_transport::Publisher aerial_map_trav_viz_pub_;

    // Subs
    ros::Subscriber aerial_map_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber goal_sub_;
    ros::Subscriber reachability_sub_;
    std::vector<ros::Subscriber> other_robot_reachability_subs_;

    // Timers
    ros::Timer timeout_timer_;
    
    // Action server
    actionlib::SimpleActionServer<spomp::GlobalNavigateAction> global_navigate_as_;

    // Objects
    Global global_;

    uint64_t last_map_stamp_{0};

    Eigen::Vector2f last_goal_{0, 0};
    bool using_action_server_{false};

    // Config related
    // Static because read in static functions
    static std::string odom_frame_;
    static std::string map_frame_;
    static std::vector<std::string> other_robot_list_;
    static std::string this_robot_;
};

} // namespace spomp
