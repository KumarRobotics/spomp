#include <gtest/gtest.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "spomp/aerial_map.h"

namespace spomp {

TEST(aerial_map_infer, test_update_reachability) {
  std::unique_ptr<AerialMap> am(new AerialMapInfer({}, {}, 8));

  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  MapReferenceFrame mrf{2, {-24.1119060516, 62.8522758484}, {}};
  mrf.setMapSizeFrom(map_img);

  am->updateMap(map_img, {}, mrf);

  Reachability reach(0, {AngularProj::StartFinish{0, 2*pi}, 100});
  reach.getScan().setConstant(5);
  reach.getScan().head<50>().setConstant(10);
  reach.getIsObs().head<50>().setConstant(1);

  am->updateLocalReachability(reach);

  cv::imwrite("spomp_aerial_map.png", am->viz());
}

TEST(aerial_map_infer, test_model_fit_stable) {
  AerialMapInfer::Params am_p;
  am_p.inference_thread_period_ms = 10;
  std::unique_ptr<AerialMap> am(new AerialMapInfer(am_p, {}, 8));
  // Does it explode if no map
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  MapReferenceFrame mrf{2, {-24.1119060516, 62.8522758484}, {}};
  mrf.setMapSizeFrom(map_img);

  am->updateMap(map_img, {}, mrf);
  // Does it explode if no trav
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  Reachability reach(0, {AngularProj::StartFinish{0, 2*pi}, 100});
  reach.getScan().setConstant(10);
  reach.getScan().head<50>().setConstant(20);
  reach.getIsObs().head<50>().setConstant(1);

  am->setTravRead();
  am->updateLocalReachability(reach);
  while (!am->haveNewTrav()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  cv::imwrite("spomp_aerial_map_infer.png", am->viz());
}

TEST(aerial_map_infer, test_map_resize) {
  AerialMapInfer::Params am_p;
  am_p.inference_thread_period_ms = 10;
  std::unique_ptr<AerialMap> am(new AerialMapInfer(am_p, {}, 8));

  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  MapReferenceFrame mrf{2, {0, 0}, {}};
  mrf.setMapSizeFrom(map_img);

  MapReferenceFrame mrf_sub = mrf;
  mrf_sub.size /= 2;
  Eigen::Vector2f mrf_sub_ul_in_mrf = mrf.world2img(
      mrf_sub.img2world({0, 0}));
  auto mrf_sub_rect = cv::Rect(cv::Point(mrf_sub_ul_in_mrf[0], mrf_sub_ul_in_mrf[1]), 
      map_img.size()/2);

  MapReferenceFrame mrf_sub2 = mrf_sub;
  mrf_sub2.center += Eigen::Vector2f(-10, 10);
  Eigen::Vector2f mrf_sub2_ul_in_mrf = mrf.world2img(
      mrf_sub2.img2world({0, 0}));
  auto mrf_sub2_rect = cv::Rect(cv::Point(mrf_sub2_ul_in_mrf[0], mrf_sub2_ul_in_mrf[1]), 
      map_img.size()/2);

  am->updateMap(map_img(mrf_sub_rect), {}, mrf_sub);
  Reachability reach(0, {AngularProj::StartFinish{0, 2*pi}, 100});
  reach.getScan().setConstant(10);
  reach.getScan().head<50>().setConstant(20);
  reach.getIsObs().head<50>().setConstant(1);

  am->setTravRead();
  am->updateLocalReachability(reach);
  while (!am->haveNewTrav()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  cv::imwrite("spomp_aerial_map_sub.png", am->viz());

  am->updateMap(map_img(mrf_sub2_rect), {}, mrf_sub2);
  cv::imwrite("spomp_aerial_map_sub2.png", am->viz());
  am->setTravRead();
  while (!am->haveNewTrav()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  cv::imwrite("spomp_aerial_map_sub2_fit.png", am->viz());

  am->updateMap(map_img, {}, mrf);
  cv::imwrite("spomp_aerial_map_full.png", am->viz());
}

TEST(mlp_model, test) {
  MLPModel mlp({100, 0.0001});
  // Not trained yet
  Eigen::VectorXf log_probs = mlp.infer(
      (Eigen::ArrayXXf(3,4) << 
       1,1,1,1,
       2,5,8,11,
       3,6,9,12).finished());
  ASSERT_EQ(log_probs.rows(), 0);

  // Train
  mlp.fit(
      (Eigen::ArrayXXf(3,4) << 
       1,1,1,1,
       2,5,8,11,
       3,6,9,12).finished(),
      (Eigen::VectorXi(4) << 0,1,0,1).finished()
      );

  log_probs = mlp.infer(
      (Eigen::ArrayXXf(3,4) << 
       1,1,1,1,
       2,5,8,11,
       3,6,9,12).finished());
  ASSERT_TRUE(std::exp(log_probs[0]) < 0.1);
  ASSERT_TRUE(std::exp(log_probs[1]) > 0.9);
  ASSERT_TRUE(std::exp(log_probs[2]) < 0.1);
  ASSERT_TRUE(std::exp(log_probs[3]) > 0.9);

  // Retrain
  mlp.fit(
      (Eigen::ArrayXXf(3,4) << 
       1,1,1,1,
       2,5,8,11,
       3,6,9,12).finished(),
      (Eigen::VectorXi(4) << 1,0,1,0).finished()
      );

  log_probs = mlp.infer(
      (Eigen::ArrayXXf(3,4) << 
       1,1,1,1,
       2,5,8,11,
       3,6,9,12).finished());
  ASSERT_TRUE(std::exp(log_probs[0]) > 0.9);
  ASSERT_TRUE(std::exp(log_probs[1]) < 0.1);
  ASSERT_TRUE(std::exp(log_probs[2]) > 0.9);
  ASSERT_TRUE(std::exp(log_probs[3]) < 0.1);
}

} // namespace spomp
