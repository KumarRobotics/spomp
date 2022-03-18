#include "spomp/mapper.h"
#include "spomp/utils.h"

namespace spomp {

Mapper::Mapper(const Params& m_p, const PoseGraph::Params& pg_p, 
    const MetricMap::Params& mm_p) 
{
  pose_graph_thread_ = std::thread(PoseGraphThread(*this, {pg_p}));
  map_thread_ = std::thread(MapThread(*this, {mm_p}));
}

Mapper::~Mapper() {
  exit_threads_flag_ = true;
  pose_graph_thread_.join();
  map_thread_.join();
}

void Mapper::addKeyframe(const Keyframe& k) {
  if ((k.getPose().inverse() * last_keyframe_pose_).translation().norm() > 
      params_.dist_between_keyframes_m) 
  {
    Keyframe::setIntrinsics(params_.pano_v_fov_rad, k.getSize());
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

void Mapper::addSemantics(const StampedSemantics& s) {
  std::scoped_lock lock(semantics_input_.mtx);
  semantics_input_.sem_panos.emplace_back(s);
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
  parse_semantics_buffer_t_ = tm.get("PG_parse_semantics_buffer");
  update_keyframes_t_ = tm.get("PG_update_keyframes");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    parseBuffer(); 
    parseSemanticsBuffer();
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

void Mapper::PoseGraphThread::parseSemanticsBuffer() {
  parse_semantics_buffer_t_->start();

  std::scoped_lock sem_lock(mapper_.semantics_input_.mtx);

  // We don't need a lock on keyframes for reading, since this thread is the only
  // thread that could add keyframes, and stamps are constant
  for (auto sem_it = mapper_.semantics_input_.sem_panos.begin();
       sem_it != mapper_.semantics_input_.sem_panos.end();) {
    auto key_it = mapper_.keyframes_.frames.find(sem_it->stamp);
    if (key_it != mapper_.keyframes_.frames.end()) {
      // We have a matching keyframe
      {
        std::unique_lock key_lock(mapper_.keyframes_.mtx);
        key_it->second.setSem(sem_it->pano);
      }
      // Advance by removing from buffer
      sem_it = mapper_.semantics_input_.sem_panos.erase(sem_it);
    } else if (mapper_.keyframes_.frames.size() > 0) {
      if (sem_it->stamp < mapper_.keyframes_.frames.rbegin()->second.getStamp()) {
        // The sem pano is older than newest images in keyframe list
        // We assume that semantic panos will always arrive later
        sem_it = mapper_.semantics_input_.sem_panos.erase(sem_it);
      } else {
        ++sem_it;
      }
    } else {
      // No keyframes yet, so no point trying to match
      break;
    }
  }

  parse_semantics_buffer_t_->end();
}

void Mapper::PoseGraphThread::updateKeyframes() {
  update_keyframes_t_->start();
  std::unique_lock key_lock(mapper_.keyframes_.mtx);

  for (auto& key : mapper_.keyframes_.frames) {
    auto new_pose = pg_.getPoseAtTime(key.first);
    if (new_pose) {
      key.second.setPose(*new_pose);
      key.second.setOptimized();
    }
  }
  mapper_.keyframes_.odom_corr = pg_.getOdomCorrection();

  update_keyframes_t_->end();
}

/*********************************************************
 * MAP THREAD
 *********************************************************/
bool Mapper::MapThread::operator()() {
  auto& tm = TimerManager::getGlobal(true);
  get_keyframes_to_compute_t_ = tm.get("M_get_keyframes_to_compute");
  resize_map_t_ = tm.get("M_resize_map");
  update_map_t_ = tm.get("M_update_map");
  export_map_t_ = tm.get("M_export_map");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!mapper_.exit_threads_flag_) {
    auto keyframes_to_compute = getKeyframesToCompute();
    if (keyframes_to_compute.size() > 0) {
      resizeMap(keyframes_to_compute);
      updateMap(keyframes_to_compute);
      exportMap();
    }

    // Sleep until next loop
    next += milliseconds(mapper_.params_.map_thread_period_ms);
    if (next < steady_clock::now()) {
      next = steady_clock::now();
    } else {
      std::this_thread::sleep_until(next);
    }
  }

  return true;
}

std::vector<Keyframe> Mapper::MapThread::getKeyframesToCompute() {
  get_keyframes_to_compute_t_->start();
  // Shared lock because we are updating map pose in keyframe, but we only ever do
  // that in this thread, so still safe to be "read-only"
  std::shared_lock lock(mapper_.keyframes_.mtx);

  std::vector<Keyframe> keyframes_to_compute;
  bool rebuild_map = false;
  for (auto& frame : mapper_.keyframes_.frames) {
    if (map_.needsMapUpdate(frame.second) && frame.second.isOptimized() &&
        (frame.second.haveSem() || !mapper_.params_.require_sem)) {
      if (frame.second.inMap()) {
        rebuild_map = true;
        break;
      }
      frame.second.updateMapPose();

      // This is a copy, but not too bad since the big data stuff is
      // in cv::Mat, which is essentially a shared_ptr
      keyframes_to_compute.push_back(frame.second);
    }
  }

  if (rebuild_map) {
    keyframes_to_compute.clear();
    map_.clear();
    
    // Add all frames
    for (auto& frame : mapper_.keyframes_.frames) {
      if (frame.second.isOptimized()) {
        frame.second.updateMapPose();
        keyframes_to_compute.push_back(frame.second);
      }
    }
  }

  get_keyframes_to_compute_t_->end();
  return keyframes_to_compute;
}

void Mapper::MapThread::resizeMap(const std::vector<Keyframe>& frames) {
  resize_map_t_->start();

  Eigen::Vector2d min = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d max = Eigen::Vector2d::Constant(std::numeric_limits<double>::lowest());
  for (auto& frame : frames) {
    Eigen::Vector3d loc = frame.getPose().translation();
    min = min.cwiseMin(loc.head<2>());
    max = max.cwiseMax(loc.head<2>());
  }
  if (frames.size() > 0) {
    map_.resizeToBounds(min, max);
  }

  resize_map_t_->end();
}

void Mapper::MapThread::updateMap(const std::vector<Keyframe>& frames) {
  update_map_t_->start();
  for (const auto& frame : frames) {
    map_.addCloud(frame.getPointCloud(), frame.getStamp());
  }
  update_map_t_->end();
}

void Mapper::MapThread::exportMap() {
  export_map_t_->start();
  {
    std::scoped_lock<std::mutex> lock(mapper_.map_messages_.mtx);
    mapper_.map_messages_.grid_map = map_.exportROSMsg();
  }
  export_map_t_->end();
}

} // namespace spomp
