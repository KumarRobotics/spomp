#include "spomp/mapper.h"

namespace spomp {

Mapper::Mapper(const Params& m_p, const PoseGraph::Params& pg_p) {
  pose_graph_thread_ = std::thread(PoseGraphThread(*this, {pg_p}));
}

Mapper::~Mapper() {
  exit_threads_flag_ = true;
  pose_graph_thread_.join();
}

void Mapper::addKeyframe(const Keyframe& k) {
  std::scoped_lock lock(keyframe_input_.mtx);
  keyframe_input_.frames.emplace_back(std::make_unique<Keyframe>(k));
}

void Mapper::addPrior(const StampedPrior& p) {
  std::scoped_lock lock(prior_input_.mtx);
  prior_input_.priors.emplace_back(std::make_unique<StampedPrior>(p));
}

/*********************************************************
 * POSE GRAPH THREAD
 *********************************************************/
bool Mapper::PoseGraphThread::operator()() {
  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    

    // Sleep until next loop
    next += milliseconds(mapper_.params_.pgo_thread_period_ms);
    if (next < steady_clock::now()) {
      next = steady_clock::now();
    } else {
      std::this_thread::sleep_until(next);
    }
  }

  return true;
}

} // namespace spomp
