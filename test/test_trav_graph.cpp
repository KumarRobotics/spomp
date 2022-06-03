#include <gtest/gtest.h>

#include "spomp/trav_graph.h"

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

  g.addEdge({n1, n4});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 3);

  g.addEdge({n0, n4});
  path = g.getPath(n0, n4);
  ASSERT_EQ(path.size(), 2);
}

} // namespace spomp
