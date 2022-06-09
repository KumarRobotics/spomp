#include "spomp/global.h"

namespace spomp {

Global::Global(const TravMap::Params& tm_p, const WaypointManager::Params& wm_p) : 
  map_(tm_p), waypoint_manager_(wm_p) {}

bool Global::setGoal(const Eigen::Vector3f& goal) {
  auto pos = waypoint_manager_.getPos();
  if (!pos) {
    // Have not yet received starting location
    return false;
  }

  auto path = map_.getPath(*pos, goal.head<2>());
  if (path.size() < 1) {
    // Cannot find path
    return false;
  }

  waypoint_manager_.setPath(path);
  return true;
}

} // namespace spomp
