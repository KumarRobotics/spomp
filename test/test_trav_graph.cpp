#include <gtest/gtest.h>

#include "spomp/trav_graph.h"

namespace spomp {

TEST(trav_graph, test_graph_search) {
  TravGraph g;
  g.addNode({{0, 0}});
  g.addNode({{1, 0}});
  g.addNode({{2, 0}});
  g.addNode({{3, 0}});
  g.addNode({{4, 1}});

  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);

  auto path = g.getPath(0, 4);
  // No path found
  ASSERT_EQ(path.size(), 0);

  g.addEdge(3, 4);
  path = g.getPath(0, 4);
  ASSERT_EQ(path.size(), 5);
}

} // namespace spomp
