#include <gtest/gtest.h>

#include "spomp/trav_graph.h"
#include "spomp/trav_map.h"

namespace spomp {

TEST(trav_graph, test_graph_search) {
  TravGraph g;
  auto* n0 = g.addNode({{0, 0}});
  auto* n1 = g.addNode({{1, 0}});
  auto* n2 = g.addNode({{2, 0}});
  auto* n3 = g.addNode({{3, 0}});
  auto* n4 = g.addNode({{4, 1}});

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
}

} // namespace spomp
