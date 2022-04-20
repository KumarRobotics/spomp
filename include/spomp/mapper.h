#pragma once

#include <thread>
#include "spomp/pose_graph.h"

namespace spomp {

class Mapper {
  public:
    Mapper();

  private:
    std::thread pose_graph_thread_;
    class PoseGraphThread {
      public:
        PoseGraphThread(Mapper& m, const PoseGraph& p) : 
          mapper_(m), pg_(p) {}

        bool operator()();

      private:
        // Reference back to parent
        Mapper& mapper_;

        PoseGraph pg_;
    };
};

} // namespace spomp
