#pragma once

#include <thread>
#include <shared_mutex>
#include "spomp/pose_graph.h"

namespace spomp {

class Mapper {
  public:
    struct Params {
      int pgo_thread_period_ms = 1000;
    };
    Mapper(const Params& m_p, const PoseGraph::Params& pg_p);

    struct Keyframe {
      long stamp = 0;
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    };
    void addKeyframe(const Keyframe& k);

    struct StampedPrior {
      long stamp = 0;
      PoseGraph::Prior2D prior{};
    };
    void addPrior(const StampedPrior& p);

    ~Mapper();

  private:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_{};

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    struct Keyframes {
      // Shared mutex here since a lot of places read only
      std::shared_mutex mtx;
      std::map<long, std::unique_ptr<Keyframe>> frames;
    } keyframes_;

    struct KeyframeInput {
      std::mutex mtx;
      std::list<std::unique_ptr<Keyframe>> frames;
    } keyframe_input_;

    struct PriorInput {
      std::mutex mtx;
      std::list<std::unique_ptr<StampedPrior>> priors;
    } prior_input_;

    /*********************************************************
     * THREADS
     *********************************************************/
    std::atomic<bool> exit_threads_flag_ = false;

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
