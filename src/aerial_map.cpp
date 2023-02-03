#include <opencv2/imgproc/imgproc.hpp>
#include "spomp/aerial_map.h"

namespace spomp {

/*********************************************************
 * AERIAL MAP PRIOR
 *********************************************************/
void AerialMapPrior::updateMap(const cv::Mat& sem_map, 
    const std::vector<cv::Mat>& dm, const MapReferenceFrame& mrf,
    const cv::Mat& color_map)
{
  // Do deep copies
  dist_maps_.clear();
  for (const auto& dist_map : dm) {
    dist_maps_.push_back(dist_map.clone());
  }
  map_ = sem_map.clone();
  map_ref_frame_ = mrf;
}

AerialMap::EdgeInfo AerialMapPrior::traceEdge(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2)
{
  auto img_pt1 = map_ref_frame_.world2img(n1);
  auto img_pt2 = map_ref_frame_.world2img(n2);
  float dist = (img_pt1 - img_pt2).norm();
  Eigen::Vector2f dir = (img_pt2 - img_pt1).normalized();

  int worst_cls = 0;
  float worst_dist = std::numeric_limits<float>::infinity();

  for (float cur_dist=0; cur_dist<dist; cur_dist+=0.5) {
    Eigen::Vector2f sample_pt = img_pt1 + dir*cur_dist;
    auto t_cls = map_.at<uint8_t>(cv::Point(sample_pt[0], sample_pt[1]));
    float dist = 0;
    if (t_cls < TravGraph::Edge::MAX_TERRAIN) {
      dist = dist_maps_[t_cls].at<float>(cv::Point(sample_pt[0], sample_pt[1]));
      //std::cout << sample_pt.transpose() << std::endl;
      //std::cout << dist << std::endl;
    
      if (t_cls > worst_cls) {
        worst_cls = t_cls;
        worst_dist = dist;
      } else if (dist < worst_dist && t_cls == worst_cls) {
        worst_dist = dist;
      }
    } else if (t_cls == TravGraph::Edge::MAX_TERRAIN) {
      worst_cls = TravGraph::Edge::MAX_TERRAIN;
      worst_dist = 0;
      // Can't get any worse than this
      break;
    }
    // Ignore unknwon
  } 
  float cost = 1/(worst_dist/map_ref_frame_.res + 0.01);
  float length = dist/map_ref_frame_.res;

  // We can scale cost however
  // Ideally, want it in range (0, 1)
  return {worst_cls, (length + std::min<float>(10*cost, 100)*length)/50};
}

/*********************************************************
 * AERIAL MAP INFER
 *********************************************************/
int AerialMapInfer::feature_size_{0};

AerialMapInfer::AerialMapInfer(const Params& p, 
    const MLPModel::Params& mlp_p, int n_cls) : 
  params_(p)
{
  auto& tm = TimerManager::getGlobal(true);
  update_reachability_t_ = tm.get("AM_update_reachability");

  // N classes + RGB + elevation
  feature_size_ = n_cls + 3 + 1;
  inference_thread_ = std::thread(InferenceThread(*this, mlp_p));
}

AerialMapInfer::~AerialMapInfer() {
  exit_threads_flag_ = true;
  inference_thread_.join();
}

void AerialMapInfer::updateMap(const cv::Mat& sem_map, 
    const std::vector<cv::Mat>& dm, const MapReferenceFrame& mrf,
    const cv::Mat& color_map)
{
  {
    std::scoped_lock lock(feature_map_.mtx);

    // We clone all of these mats since there are multiple threads involved
    // here, so safer to not have to worry about changes upstream
    if (sem_map.channels() == 1) {
      feature_map_.sem_map = sem_map.clone();
    } else {
      // In the event we receive a 3 channel image, assume all channels are
      // the same
      cv::cvtColor(sem_map, feature_map_.sem_map, cv::COLOR_BGR2GRAY);
    }
    feature_map_.dist_maps.clear();
    for (const auto& dist_map : dm) {
      feature_map_.dist_maps.push_back(dist_map.clone());
    }
    feature_map_.color_map = color_map.clone();
  }

  // Location of upper left corner of old map in new map
  Eigen::Vector2f ul_old_in_new = mrf.world2img(
      map_ref_frame_.img2world({0, 0}));
  cv::Rect old_roi(cv::Point(ul_old_in_new[0], ul_old_in_new[1]), 
      cv::Size(map_ref_frame_.size[0], map_ref_frame_.size[1]));
  cv::Rect new_roi({}, sem_map.size());

  auto intersect = new_roi & old_roi;
  auto intersect_old_frame = intersect;
  intersect_old_frame -= old_roi.tl();

  {
    std::scoped_lock lock(reachability_map_.mtx);
    cv::Mat old_reach_map = reachability_map_.map.clone();
    reachability_map_.map = cv::Mat::zeros(sem_map.size(), CV_16SC1);
    if (!intersect.empty()) {
      old_reach_map(intersect_old_frame).copyTo(reachability_map_.map(intersect));
    }
  }
  {
    std::scoped_lock lock(prob_map_.mtx);
    cv::Mat old_prob_map = prob_map_.map.clone();
    prob_map_.map = cv::Mat::zeros(sem_map.size(), CV_32FC1);
    if (!intersect.empty()) {
      old_prob_map(intersect_old_frame).copyTo(prob_map_.map(intersect));
    }
  }

  map_ref_frame_ = mrf;
}

void AerialMapInfer::updateLocalReachability(const Reachability& reach) {
  update_reachability_t_->start();

  Eigen::Vector2f img_center = 
    map_ref_frame_.world2img(reach.getPose().translation());

  Eigen::VectorXf thetas = reach.getProj().getAngles();
  thetas.array() += Eigen::Rotation2Df(reach.getPose().rotation()).angle();

  cv::Mat trav_map_delta = cv::Mat::zeros(/*rows*/ map_ref_frame_.size[1], 
      /*cols*/ map_ref_frame_.size[0], CV_16SC1);

  for (int i=0; i<thetas.size(); ++i) {
    Eigen::Vector2f ray_dir(-sin(thetas[i]), -cos(thetas[i]));
    auto ray_info = reach.getObsAtInd(i);
    for (float range=0; range<ray_info.range; range+=1./map_ref_frame_.res) {
      Eigen::Vector2f pt = ray_dir*range*map_ref_frame_.res + img_center;
      if (map_ref_frame_.imgPointInMap(pt)) {
        trav_map_delta.at<int16_t>(cv::Point(pt[0], pt[1])) = 1;
      }
    }

    if (ray_info.is_obs) {
      for (float range=ray_info.range; range<ray_info.range + params_.not_trav_range_m; 
          range+=1./map_ref_frame_.res) 
      {
        Eigen::Vector2f pt = ray_dir*range*map_ref_frame_.res + img_center;
        if (map_ref_frame_.imgPointInMap(pt)) {
          trav_map_delta.at<int16_t>(cv::Point(pt[0], pt[1])) = -1;
        }
      }
    }
  }

  {
    std::scoped_lock lock(reachability_map_.mtx);
    reachability_map_.map += trav_map_delta;
  }

  update_reachability_t_->end();
}

AerialMap::EdgeInfo AerialMapInfer::traceEdge(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2)
{
  auto img_pt1 = map_ref_frame_.world2img(n1);
  auto img_pt2 = map_ref_frame_.world2img(n2);
  float dist = (img_pt1 - img_pt2).norm();
  Eigen::Vector2f dir = (img_pt2 - img_pt1).normalized();

  std::scoped_lock lock(prob_map_.mtx);
  float total_neg_log_prob = 0;
  float num_valid = 0;
  float num_invalid = 0;
  for (float cur_dist=0; cur_dist<dist; cur_dist+=map_ref_frame_.res) {
    Eigen::Vector2f sample_pt = img_pt1 + dir*cur_dist;
    float prob = prob_map_.map.at<float>(cv::Point(sample_pt[0], sample_pt[1]));
    if (prob != 0) {
      total_neg_log_prob += -std::log(std::max<float>(prob, 0.0001))/5;
      ++num_valid;
    } else {
      ++num_invalid;
    }
  }

  if (num_valid == 0) {
    // If everything is unknown, just use a small cost weighted by distance
    // Make small cost so that when we have real costs the cost will
    // increase, forcing a path recomputation
    total_neg_log_prob = 0.001 * dist / map_ref_frame_.res;
  } else {
    // Scale by unknown.  Otherwise if we have a mostly unknown edge,
    // it will have disproportionally low cost
    total_neg_log_prob *= (num_valid + num_invalid)/num_valid;
  }

  return {0, total_neg_log_prob};
}


cv::Mat AerialMapInfer::viz() {
  cv::Mat trav_viz;
  constexpr int new_center = std::numeric_limits<uint8_t>::max()/2;
  {
    std::scoped_lock lock(reachability_map_.mtx);
    reachability_map_.map.convertTo(trav_viz, CV_8UC1, 1, new_center);
  }

  auto center = map_ref_frame_.world2img({0,0});

  cv::cvtColor(trav_viz, trav_viz, cv::COLOR_GRAY2BGR);

  using Pixel = cv::Point3_<uint8_t>;
  trav_viz.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
    if (pixel.x <= new_center - params_.not_trav_thresh) {
      pixel.x = 0;   // b
      pixel.y = 0;   // g
      pixel.z = 255; // r
    } else if (pixel.x >= new_center + params_.trav_thresh) {
      pixel.x = 0;   // b
      pixel.y = 255; // g
      pixel.z = 0;   // r
    } else {
      // black
      pixel.x = 0;
      pixel.y = 0;
      pixel.z = 0;
    }
  });

  cv::Mat prob_viz;
  {
    std::scoped_lock lock(prob_map_.mtx);
    prob_map_.map.convertTo(prob_viz, CV_8UC1, 255, 0);
  }
  cv::applyColorMap(prob_viz, prob_viz, cv::COLORMAP_RAINBOW);

  if (prob_viz.empty()) {
    // Case where model hasn't trained yet
    trav_viz = prob_viz;
  } else {
    cv::addWeighted(trav_viz, 1, prob_viz, 0.5, 0, trav_viz);
  }

  return trav_viz;
}

/*********************************************************
 * INFERENCE THREAD
 *********************************************************/
bool AerialMapInfer::InferenceThread::operator()() {
  auto& tm = TimerManager::getGlobal(true);
  model_fit_t_ = tm.get("AM_model_fit");
  update_probability_map_t_ = tm.get("AM_update_probability_map");

  using namespace std::chrono;
  auto next = steady_clock::now();
  while (!aerial_map_.exit_threads_flag_) {
    // Do stuff
    fitModel();
    cv::Mat prob_map = updateProbabilityMap();
    {
      std::scoped_lock lock(aerial_map_.prob_map_.mtx);
      if (prob_map.size() == aerial_map_.prob_map_.map.size()) {
        aerial_map_.prob_map_.map = prob_map;
        aerial_map_.prob_map_.have_new = true;
      }
    }

    // Sleep until next loop
    next += milliseconds(aerial_map_.params_.inference_thread_period_ms);
    if (next < steady_clock::now()) {
      next = steady_clock::now();
    } else {
      std::this_thread::sleep_until(next);
    }
  }

  return true;
}

Eigen::VectorXf AerialMapInfer::InferenceThread::getFeatureAtPoint(
    const AerialMapInfer::FeatureMap& feat_map, const cv::Point& pt) 
{
  Eigen::VectorXf feat = Eigen::VectorXf::Zero(feature_size_);
  if (feat_map.dist_maps.size() > 0 && 
      feat_map.dist_maps[0].size() == feat_map.sem_map.size()) 
  {
    int cls = 0;
    for (const auto& dist_map : feat_map.dist_maps) {
      feat[cls] = dist_map.at<float>(pt);
      ++cls;
    }
  } else {
    // Fallback
    int cls = feat_map.sem_map.at<uint8_t>(pt);
    if (cls < feat.size()) {
      feat[cls] = 1;
    }
  }

  if (feat_map.color_map.size() == feat_map.sem_map.size()) {
    const auto& color = feat_map.color_map.at<cv::Vec3b>(pt);
    feat[feature_size_ - 4] = color[0];
    feat[feature_size_ - 3] = color[1];
    feat[feature_size_ - 2] = color[2];
  }

  return feat;
}

void AerialMapInfer::InferenceThread::fitModel() {
  // No more features than number of nonzero pixels in trav map
  int max_features;
  Eigen::ArrayXXf features;
  Eigen::VectorXi labels;

  int n_features = 0;
  {
    std::scoped_lock lock_r(aerial_map_.reachability_map_.mtx);
    std::scoped_lock lock_f(aerial_map_.feature_map_.mtx);

    max_features = cv::countNonZero(aerial_map_.reachability_map_.map);
    if (max_features == 0) return;
    features = Eigen::ArrayXXf(feature_size_, max_features);
    labels = Eigen::VectorXi(max_features);

    for(int i=0; i<aerial_map_.reachability_map_.map.rows; ++i) {
      for(int j=0; j<aerial_map_.reachability_map_.map.cols; ++j) {
        cv::Point pt(j, i);
        int16_t trav_map_pt = aerial_map_.reachability_map_.map.at<int16_t>(pt);
        if (trav_map_pt <= -aerial_map_.params_.not_trav_thresh) {
          features.col(n_features) = getFeatureAtPoint(
              aerial_map_.feature_map_, pt);
          labels[n_features] = 0;
          ++n_features;
        } else if (trav_map_pt >= aerial_map_.params_.trav_thresh) {
          features.col(n_features) = getFeatureAtPoint(
              aerial_map_.feature_map_, pt);
          labels[n_features] = 1;
          ++n_features;
        }
      }
    }
  }
  features.conservativeResize(Eigen::NoChange, n_features);
  labels.conservativeResize(n_features);

  model_fit_t_->start();
  model_.fit(features, labels);
  model_fit_t_->end();
}

cv::Mat AerialMapInfer::InferenceThread::updateProbabilityMap() {
  cv::Mat prob_map;
  if (!model_.trained()) return prob_map;

  Eigen::ArrayXXf features;
  int feat_cnt = 0;
  std::vector<float*> prob_ptrs;
  {
    std::scoped_lock lock(aerial_map_.feature_map_.mtx);
    int n_features = cv::countNonZero(aerial_map_.feature_map_.sem_map < 255);
    if (n_features == 0) return prob_map;
        
    prob_map = cv::Mat::zeros(aerial_map_.feature_map_.sem_map.size(), CV_32FC1);
    features = Eigen::ArrayXXf(feature_size_, n_features);
    prob_ptrs.resize(n_features);

    for(int i=0; i<prob_map.rows; ++i) {
      for(int j=0; j<prob_map.cols; ++j) {
        cv::Point pt(j, i);

        if (aerial_map_.feature_map_.sem_map.at<uint8_t>(pt) < 255) {
          prob_ptrs[feat_cnt] = prob_map.ptr<float>(i, j);
          features.col(feat_cnt) = getFeatureAtPoint(
              aerial_map_.feature_map_, pt);
          ++feat_cnt;
        }
      }
    }
  }

  update_probability_map_t_->start();
  Eigen::VectorXf pred_log_probs = model_.infer(features);
  update_probability_map_t_->end();

  for (int i=0; i<pred_log_probs.size(); ++i) {
    *(prob_ptrs[i]) = std::exp(pred_log_probs[i]);
  }

  return prob_map;
}

} // namespace spomp
