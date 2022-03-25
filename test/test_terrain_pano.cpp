#include <gtest/gtest.h>
#include "spomp/terrain_pano.h"
#include <iostream>

namespace spomp {

TEST(terrain_pano, test_fill_holes) {
  Eigen::ArrayXXf pano = Eigen::ArrayXXf::Ones(3, 100);
  pano.row(0).setLinSpaced(100, 1, 100);
  Eigen::ArrayXXf pano_copy = pano;

  // Create hole
  pano.block<3, 5>(0, 10) = 0;
  ASSERT_TRUE((pano != pano_copy).any());

  TerrainPano tp({});
  tp.fillHoles(pano);

  ASSERT_TRUE(((pano - pano_copy).abs() < 0.001).all());

  // Test wrap-around
  pano = 1;
  for (int i = 91; i<100; i++) {
    pano(0, i) = i - 90;
  }
  for (int i = 0; i<10; i++) {
    pano(0, i) = 10+i;
  }
  pano_copy = pano;
  pano.block<3, 8>(0, 92) = 0;
  pano.block<3, 9>(0, 0) = 0;
  ASSERT_TRUE((pano != pano_copy).any());
  tp.fillHoles(pano);
  ASSERT_TRUE(((pano - pano_copy).abs() < 0.001).all());
}

} // namespace spomp
