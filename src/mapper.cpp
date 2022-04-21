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

std::vector<Eigen::Isometry3d> Mapper::getGraph() { 
  std::vector<Eigen::Isometry3d> poses;
  std::shared_lock key_lock(keyframes_.mtx);
  poses.reserve(keyframes_.frames.size());

  for (const auto& frame : keyframes_.frames) {
    poses.emplace_back(frame.second->pose);
  }

  return poses;
}

/*********************************************************
 * POSE GRAPH THREAD
 *********************************************************/
bool Mapper::PoseGraphThread::operator()() {
  auto& tm = TimerManager::getGlobal(true);
  parse_buffer_t_ = tm.get("PG_parse_buffer");
  graph_update_t_ = tm.get("PG_graph_update");
  update_keyframes_t_ = tm.get("PG_update_keyframes");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    parseBuffer(); 
    graph_update_t_->start();
    pg_.update();
    graph_update_t_->end();
    updateKeyframes();

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

void Mapper::PoseGraphThread::parseBuffer() {
  parse_buffer_t_->start();
  {
    // Service keyframe buffer
    std::scoped_lock lock(mapper_.keyframe_input_.mtx);

    Eigen::Isometry3d pg_pose;
    for (auto& frame : mapper_.keyframe_input_.frames) {
      auto ind = pg_.addNode(frame->stamp, frame->pose);
      // PoseGraph updates pose on addition based on relative motion
      pg_pose = pg_.getPoseAtIndex(ind);
      frame->pose = pg_pose;

      std::unique_lock key_lock(mapper_.keyframes_.mtx);
      mapper_.keyframes_.frames.insert({frame->stamp, std::move(frame)});
    }

    mapper_.keyframe_input_.frames.clear();
  }
  {
    // Service prior buffer
    std::scoped_lock lock(mapper_.prior_input_.mtx);

    for (auto& prior : mapper_.prior_input_.priors) {
      pg_.addPrior(prior->stamp, std::move(prior->prior));
    }

    mapper_.prior_input_.priors.clear();
  }
  parse_buffer_t_->end();
}

void Mapper::PoseGraphThread::updateKeyframes() {
  update_keyframes_t_->start();
  std::unique_lock key_lock(mapper_.keyframes_.mtx);

  for (auto& key : mapper_.keyframes_.frames) {
    auto new_pose = pg_.getPoseAtTime(key.first);
    if (new_pose) {
      key.second->pose = *new_pose;
    }
  }
  update_keyframes_t_->end();
}

} // namespace spomp
