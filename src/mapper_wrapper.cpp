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
  ros::spin();
}

} // namespace spomp
