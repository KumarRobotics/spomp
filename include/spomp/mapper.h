#pragma once

#include <thread>
#include <shared_mutex>
#include "spomp/pose_graph.h"
#include "spomp/timer.h"
#include "spomp/keyframe.h"

namespace spomp {

class Mapper {
  public:
    struct Params {
      int pgo_thread_period_ms = 1000;
      bool correct_odom_per_frame = true;
      float dist_between_keyframes_m = 5;
    };
    Mapper(const Params& m_p, const PoseGraph::Params& pg_p);

    void addKeyframe(const Keyframe& k);

    struct StampedPrior {
      long stamp = 0;
      PoseGraph::Prior2D prior{};
    };
    void addPrior(const StampedPrior& p);

    //! @return Vector of keyframe poses oldest to most recent
    std::vector<Eigen::Isometry3d> getGraph();

    //! @return Corrective pose to transform odom into map frame
    Eigen::Isometry3d getOdomCorrection();

    //! @return Timestamp of most recent keyframe
    long stamp();

    ~Mapper();

  private:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_{};

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    StampedPrior last_prior_{};
    Eigen::Isometry3d last_keyframe_pose_{Eigen::Isometry3d::Identity()};

    struct Keyframes {
      // Shared mutex here since a lot of places read only
      std::shared_mutex mtx;
      std::map<long, Keyframe> frames;
      Eigen::Isometry3d odom_corr = Eigen::Isometry3d::Identity();
    } keyframes_;

    struct KeyframeInput {
      std::mutex mtx;
      std::list<Keyframe> frames;
    } keyframe_input_;

    struct PriorInput {
      std::mutex mtx;
      std::list<StampedPrior> priors;
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
        void parseBuffer();

        void updateKeyframes();

        // Reference back to parent
        Mapper& mapper_;

        PoseGraph pg_;

        // Timers
        Timer* parse_buffer_t_{};
        Timer* update_keyframes_t_{};
    };
};

} // namespace spomp
