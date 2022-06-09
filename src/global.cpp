#include "spomp/global.h"

namespace spomp {

Global::Global(const TravMap::Params& tm_p) : 
  map_(tm_p) {}

bool Global::setGoal(const Eigen::Vector3f& goal) {
  return false;
}

std::optional<Eigen::Vector2f> Global::getNextWaypoint(
    const Eigen::Isometry3f& state) 
{
  return {};
}

} // namespace spomp
