#include <gtest/gtest.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "spomp/trav_graph.h"
#include "spomp/trav_map.h"
#include "spomp/utils.h"

namespace spomp {

TEST(trav_graph, test_graph_search) {
  TravGraph g({});
  TravGraph::Node* n0 = g.addNode({{0, 0}});
  TravGraph::Node* n1 = g.addNode({{1, 0}});
  TravGraph::Node* n2 = g.addNode({{2, 0}});
  TravGraph::Node* n3 = g.addNode({{3, 0}});
  TravGraph::Node* n4 = g.addNode({{4, 1}});

  g.addEdge({n0, n1, 1});
  g.addEdge({n1, n2, 1});
  g.addEdge({n2, n3, 1});

  auto path = g.getPath(n0, n4);
  // No path found
  ASSERT_EQ(path.size(), 0);

  g.addEdge({n3, n4, 1});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 5);
  ASSERT_FLOAT_EQ(path.back()->cost, 3 + std::sqrt(2));

  g.addEdge({n1, n4, 1});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 3);
  ASSERT_FLOAT_EQ(path.back()->cost, 1 + std::sqrt(9 + 1));

  g.addEdge({n0, n4, 1});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 2);
  ASSERT_FLOAT_EQ(path.back()->cost, std::sqrt(16 + 1));

  auto edge = n0->getEdgeToNode(n4);
  ASSERT_TRUE(edge);
  ASSERT_FLOAT_EQ(edge->length, std::sqrt(16 + 1));
  edge = n0->getEdgeToNode(n1);
  ASSERT_TRUE(edge);
  ASSERT_FLOAT_EQ(edge->length, 1);
  edge = n0->getEdgeToNode(n2);
  ASSERT_TRUE(!edge);
}

TEST(reachability, test_reach) {
  // Initial setup
  Reachability reach{0, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), 
    Eigen::Isometry2f::Identity()};
  reach.setScan(Eigen::VectorXf::Ones(360)*10);
  Eigen::VectorXi is_obs = Eigen::VectorXi::Ones(360);
  is_obs.head<180>().setConstant(0);
  reach.setIsObs(is_obs);

  // Inside
  ASSERT_EQ(reach.analyzeEdge({0, 0}, {5, 0}, {0.2}), Reachability::TRAV);
  ASSERT_EQ(reach.analyzeEdge({-5, 0}, {5, 0}, {0.2}), Reachability::TRAV);
  // Outside
  ASSERT_EQ(reach.analyzeEdge({0, 0}, {0, -15}, {0.2}), Reachability::NOT_TRAV);
  ASSERT_EQ(reach.analyzeEdge({0, 0}, {0, 15}, {0.2}), Reachability::UNKNOWN);
  ASSERT_EQ(reach.analyzeEdge({5, 0}, {5, 15}, {0.2}), Reachability::UNKNOWN);
  // Both outside (have to be unknown)
  ASSERT_EQ(reach.analyzeEdge({5, -15}, {5, 15}, {0.2}), Reachability::UNKNOWN);
}

TEST(trav_map, test_transforms) {
  TravMap::Params m_p;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config_dynamic.yaml";
  TravMap m(m_p, {}, {}, {});
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  m.updateMap(map_img, {-24.1119060516, 62.8522758484});

  Eigen::Vector2f im_pt{100, 200};
  Eigen::Vector2f world_pt = m.getMapReferenceFrame().img2world(im_pt);
  ASSERT_FLOAT_EQ((im_pt - m.getMapReferenceFrame().world2img(world_pt)).norm(), 0);
}

TEST(trav_map, test_uniform_sampling) {
  TravMap::Params m_p;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config_dynamic.yaml";
  m_p.no_max_terrain_in_graph = false;
  m_p.uniform_node_sampling = true;
  TravMap m(m_p, {}, {}, {});
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  m.updateMap(map_img, {-24.1119060516, 62.8522758484});

  // save
  cv::imwrite("spomp_uniform_trav_map.png", m.viz());
  cv::imwrite("spomp_uniform_viz_map.png", m.viz_visibility());
}

TEST(trav_map, test_map_graph_search) {
  TravMap::Params m_p;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config_dynamic.yaml";
  m_p.no_max_terrain_in_graph = false;
  TravMap m(m_p, {}, {}, {});
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  m.updateMap(map_img, {-24.1119060516, 62.8522758484});

  // save
  cv::imwrite("spomp_trav_map.png", m.viz());
  cv::imwrite("spomp_viz_map.png", m.viz_visibility());

  // test path
  // road onto grass
  auto path = m.getPath(m.getMapReferenceFrame().img2world({90, 244}),
                        m.getMapReferenceFrame().img2world({225, 127}));
  ASSERT_TRUE(path.size() > 0);
  ASSERT_TRUE(path.back()->cost < std::pow(100, 3));
  ASSERT_TRUE(path.back()->cost > std::pow(100, 1));

  // stay on road
  path = m.getPath(m.getMapReferenceFrame().img2world({392, 135}), {0, 0});
  ASSERT_TRUE(path.size() > 0);
  ASSERT_TRUE(path.back()->cost < std::pow(100, 1));
}

TEST(trav_map, test_static_map) {
  TravMap::Params m_p;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config.yaml";
  TravMap m(m_p, {}, {}, {});

  // save
  cv::imwrite("spomp_static_trav_map.png", m.viz());
  cv::imwrite("spomp_static_viz_map.png", m.viz_visibility());
}

TEST(trav_map, test_update_map) {
  TravMap::Params m_p;
  TravGraph::Params g_p;
  g_p.reach_node_max_dist_m = 3;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config_dynamic.yaml";
  TravMap m(m_p, g_p, {}, {});
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  m.updateMap(map_img, {-24.1119060516, 62.8522758484});

  Reachability reach{0, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), 
    Eigen::Isometry2f::Identity()};

  Eigen::VectorXf scan = Eigen::VectorXf::Ones(360)*100;
  scan.tail<180>().setConstant(0);
  reach.setScan(scan);

  reach.setIsObs(Eigen::VectorXi::Ones(360));
  m.updateLocalReachability(reach);
  ASSERT_EQ(m.getReachBufSize(), 0);

  cv::imwrite("spomp_trav_updated_map.png", m.viz());
}

TEST(trav_map, test_reach_before_map) {
  TravMap::Params m_p;
  TravGraph::Params g_p;
  g_p.reach_node_max_dist_m = 3;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config_dynamic.yaml";
  TravMap m(m_p, g_p, {}, {});
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") +
                               "/test/map.png");

  Reachability reach{0, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360),
    Eigen::Isometry2f::Identity()};

  Eigen::VectorXf scan = Eigen::VectorXf::Ones(360)*100;
  scan.tail<180>().setConstant(0);
  reach.setScan(scan);

  reach.setIsObs(Eigen::VectorXi::Ones(360));

  ASSERT_EQ(m.getReachBufSize(), 0);
  m.updateLocalReachability(reach);
  ASSERT_EQ(m.getReachBufSize(), 1);
  m.updateMap(cv::Mat(cv::Size(10, 10), CV_8UC1), {20, 20});
  ASSERT_EQ(m.getReachBufSize(), 1);

  m.updateMap(map_img, {-24.1119060516, 62.8522758484});
  ASSERT_EQ(m.getReachBufSize(), 0);

  m.updateLocalReachability(reach);
  ASSERT_EQ(m.getReachBufSize(), 0);
}

TEST(trav_map, test_update_reach) {
  TravMap::Params m_p;
  m_p.world_config_path = ros::package::getPath("semantics_manager") + "/config/test_config_dynamic.yaml";
  TravMap m(m_p, {}, {}, {});

  ASSERT_EQ(m.getReachabilityHistory().size(), 0);
  Eigen::Isometry2f pose = Eigen::Isometry2f::Identity();
  m.updateLocalReachability({0, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), pose}, 0);
  ASSERT_EQ(m.getReachabilityHistory().size(), 1);
  pose.translate(Eigen::Vector2f(1, 0));
  m.updateLocalReachability({1, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), pose}, 0);
  ASSERT_EQ(m.getReachabilityHistory().size(), 1);
  pose.translate(Eigen::Vector2f(2, 0));
  m.updateLocalReachability({2, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), pose}, 0);
  ASSERT_EQ(m.getReachabilityHistory().size(), 2);
  pose.translate(Eigen::Vector2f(-3, 0));
  m.updateLocalReachability({3, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), pose}, 0);
  ASSERT_EQ(m.getReachabilityHistory().size(), 2);
  pose.translate(Eigen::Vector2f(-2, 0));
  m.updateLocalReachability({4, AngularProj(AngularProj::StartFinish{0, 2*pi}, 360), pose}, 0);
  ASSERT_EQ(m.getReachabilityHistory().size(), 3);
}

} // namespace spomp
