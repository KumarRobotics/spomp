#include "spomp/mapper.h"
#include "spomp/utils.h"

namespace spomp {

Mapper::Mapper(const Params& m_p, const PoseGraph::Params& pg_p) {
  pose_graph_thread_ = std::thread(PoseGraphThread(*this, {pg_p}));
}

Mapper::~Mapper() {
  exit_threads_flag_ = true;
  pose_graph_thread_.join();
}

void Mapper::addKeyframe(const Keyframe& k) {
  if ((k.getPose().inverse() * last_keyframe_pose_).translation().norm() > 
      params_.dist_between_keyframes_m) 
  {
    last_keyframe_pose_ = k.getPose();
    std::scoped_lock lock(keyframe_input_.mtx);
    keyframe_input_.frames.emplace_back(k);
  }
}

void Mapper::addPrior(const StampedPrior& p) {
  last_prior_ = p;
  std::scoped_lock lock(prior_input_.mtx);
  prior_input_.priors.emplace_back(p);
}

std::vector<Eigen::Isometry3d> Mapper::getGraph() { 
  std::vector<Eigen::Isometry3d> poses;
  std::shared_lock key_lock(keyframes_.mtx);
  poses.reserve(keyframes_.frames.size());

  for (const auto& frame : keyframes_.frames) {
    poses.emplace_back(frame.second.getPose());
  }

  return poses;
}

Eigen::Isometry3d Mapper::getOdomCorrection() {
  Eigen::Isometry3d corr;
  if (!params_.correct_odom_per_frame) {
    std::shared_lock key_lock(keyframes_.mtx);
    corr = keyframes_.odom_corr;
  } else {
    corr = pose22pose3(last_prior_.prior.pose) * 
           last_prior_.prior.local_pose.inverse();
  }
  return corr;
}

long Mapper::stamp() {
  std::shared_lock key_lock(keyframes_.mtx);
  if (keyframes_.frames.size() < 1) return 0;
  return keyframes_.frames.rbegin()->first;
}

/*********************************************************
 * POSE GRAPH THREAD
 *********************************************************/
bool Mapper::PoseGraphThread::operator()() {
  auto& tm = TimerManager::getGlobal(true);
  parse_buffer_t_ = tm.get("PG_parse_buffer");
  update_keyframes_t_ = tm.get("PG_update_keyframes");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    parseBuffer(); 
    pg_.update();
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
      auto ind = pg_.addNode(frame.getStamp(), frame.getPose());
      // PoseGraph updates pose on addition based on relative motion
      pg_pose = pg_.getPoseAtIndex(ind);
      frame.setPose(pg_pose);

      std::unique_lock key_lock(mapper_.keyframes_.mtx);
      mapper_.keyframes_.frames.insert({frame.getStamp(), std::move(frame)});
    }

    mapper_.keyframe_input_.frames.clear();
  }
  {
    // Service prior buffer
    std::scoped_lock lock(mapper_.prior_input_.mtx);

    for (auto& prior : mapper_.prior_input_.priors) {
      pg_.addPrior(prior.stamp, std::move(prior.prior));
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
      key.second.setPose(*new_pose);
    }
  }
  mapper_.keyframes_.odom_corr = pg_.getOdomCorrection();

  update_keyframes_t_->end();
}

} // namespace spomp
