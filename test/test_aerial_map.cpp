#include <gtest/gtest.h>

#include "spomp/aerial_map.h"

namespace spomp {

TEST(aerial_map, test) {
  AerialMap am({});
  am.testML();
}

} // namespace spomp
