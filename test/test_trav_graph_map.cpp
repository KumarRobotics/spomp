#include <gtest/gtest.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "spomp/trav_graph.h"
#include "spomp/trav_map.h"

namespace spomp {

TEST(trav_graph, test_graph_search) {
  TravGraph g;
  TravGraph::Node* n0 = g.addNode({{0, 0}});
  TravGraph::Node* n1 = g.addNode({{1, 0}});
  TravGraph::Node* n2 = g.addNode({{2, 0}});
  TravGraph::Node* n3 = g.addNode({{3, 0}});
  TravGraph::Node* n4 = g.addNode({{4, 1}});

  g.addEdge({n0, n1});
  g.addEdge({n1, n2});
  g.addEdge({n2, n3});

  auto path = g.getPath(n0, n4);
  // No path found
  ASSERT_EQ(path.size(), 0);

  g.addEdge({n3, n4});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 5);
  ASSERT_FLOAT_EQ(path.back()->cost, 3 + std::sqrt(2));

  g.addEdge({n1, n4});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 3);
  ASSERT_FLOAT_EQ(path.back()->cost, 1 + std::sqrt(9 + 1));

  g.addEdge({n0, n4});
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

TEST(trav_map, test_transforms) {
  TravMap::Params m_p;
  m_p.terrain_types_path = ros::package::getPath("spomp") + "/config/terrain_types.yaml";
  TravMap m(m_p);
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  m.updateMap(map_img, {-24.1119060516, 62.8522758484});

  Eigen::Vector2f im_pt{100, 200};
  Eigen::Vector2f world_pt = m.img2world(im_pt);
  ASSERT_FLOAT_EQ((im_pt - m.world2img(world_pt)).norm(), 0);
}

TEST(trav_map, test) {
  TravMap::Params m_p;
  m_p.terrain_types_path = ros::package::getPath("spomp") + "/config/terrain_types.yaml";
  TravMap m(m_p);
  cv::Mat map_img = cv::imread(ros::package::getPath("spomp") + 
                               "/test/map.png");
  m.updateMap(map_img, {-24.1119060516, 62.8522758484});

  // save
  cv::imwrite("spomp_trav_map.png", m.viz());
  cv::imwrite("spomp_viz_map.png", m.viz_visibility());

  // test path
  auto path = m.getPath({38, -11}, {-107, 42});
  ASSERT_TRUE(path.size() > 0);
  ASSERT_TRUE(path.back()->cost < std::pow(100, 3));

  path = m.getPath({38, -11}, {0, 0});
  ASSERT_TRUE(path.size() > 0);
  ASSERT_TRUE(path.back()->cost < std::pow(100, 1));
}

} // namespace spomp
