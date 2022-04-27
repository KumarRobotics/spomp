#include "spomp/mapper_wrapper.h"
#include "spomp/rosutil.h"
#include <visualization_msgs/MarkerArray.h>

namespace spomp {

MapperWrapper::MapperWrapper(ros::NodeHandle& nh) : 
  nh_(nh),
  mapper_(createMapper(nh)) 
{
  auto& tm = TimerManager::getGlobal(true);
  viz_t_ = tm.get("MW_viz");

  // Publishers
  graph_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("graph_viz", 1);
}

Mapper MapperWrapper::createMapper(ros::NodeHandle& nh) {
  Mapper::Params m_params;
  PoseGraph::Params pg_params;

  return Mapper(m_params, pg_params);
}

void MapperWrapper::initialize() {
  // Subscribers
  odom_sub_ = nh_.subscribe("odom", 1, &MapperWrapper::odomCallback, this);
  global_est_sub_ = nh_.subscribe("global_est", 
      1, &MapperWrapper::globalEstCallback, this);

  ros::spin();
}

void MapperWrapper::odomCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
}

void MapperWrapper::globalEstCallback(
    const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
}

} // namespace spomp
