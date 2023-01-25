#include <gtest/gtest.h>
#include "spomp/aerial_map.h"

namespace spomp {

TEST(aerial_map, test) {
  AerialMap am({});
}

TEST(mlp_model, test) {
  MLPModel mlp({100, 0.0001});
  mlp.fit(
      (Eigen::ArrayXXf(4,3) << 
       1,2,3,
       4,5,6,
       7,8,9,
       10,11,12).finished(),
      (Eigen::VectorXi(4) << 0,1,0,1).finished()
      );

  Eigen::VectorXf log_probs = mlp.infer(
      (Eigen::ArrayXXf(4,3) << 
       1,2,3,
       4,5,6,
       7,8,9,
       10,11,12).finished());
  ASSERT_TRUE(std::exp(log_probs[0]) < 0.1);
  ASSERT_TRUE(std::exp(log_probs[1]) > 0.9);
  ASSERT_TRUE(std::exp(log_probs[2]) < 0.1);
  ASSERT_TRUE(std::exp(log_probs[3]) > 0.9);
}

} // namespace spomp
