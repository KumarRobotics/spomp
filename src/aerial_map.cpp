#include <opencv2/imgproc/imgproc.hpp>
#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p, const MLPModel::Params& mlp_p) : 
  params_(p)
{
  auto& tm = TimerManager::getGlobal(true);
  update_reachability_t_ = tm.get("AM_update_reachability");

  inference_thread_ = std::thread(InferenceThread(*this, mlp_p));
}

AerialMap::~AerialMap() {
  exit_threads_flag_ = true;
  inference_thread_.join();
}

void AerialMap::updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf) {
  if (sem_map.channels() == 1) {
    std::scoped_lock lock(feature_map_.mtx);
    feature_map_.map = sem_map;
  } else {
    // In the event we receive a 3 channel image, assume all channels are
    // the same
    std::scoped_lock lock(feature_map_.mtx);
    cv::cvtColor(sem_map, feature_map_.map, cv::COLOR_BGR2GRAY);
  }
  map_ref_frame_ = mrf;
  {
    std::scoped_lock lock(reachability_map_.mtx);
    reachability_map_.map = cv::Mat::zeros(sem_map.size(), CV_16SC1);
  }
  {
    std::scoped_lock lock(prob_map_.mtx);
    prob_map_.map = cv::Mat::zeros(sem_map.size(), CV_32FC1);
  }
}

void AerialMap::updateLocalReachability(const Reachability& reach) {
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

float AerialMap::getEdgeProb(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2)
{
  Eigen::Vector2f n1_img = map_ref_frame_.world2img(n1);
  Eigen::Vector2f n2_img = map_ref_frame_.world2img(n2);

  Eigen::Vector2f diff = n2_img - n1_img;
  Eigen::Vector2f diff_dir = diff.normalized();

  std::scoped_lock lock(prob_map_.mtx);
  float worst_prob = 1;
  for (int d=0; d<diff.norm(); d += 0.5) {
    Eigen::Vector2f pt = n1_img + diff_dir*d;
    float prob = prob_map_.map.at<float>(cv::Point(pt[0], pt[1]));
    worst_prob = std::min(prob, worst_prob);
  }

  return worst_prob;
}


cv::Mat AerialMap::viz() {
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
bool AerialMap::InferenceThread::operator()() {
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
      aerial_map_.prob_map_.map = prob_map;
      aerial_map_.prob_map_.have_new = true;
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

Eigen::VectorXf AerialMap::InferenceThread::getFeatureAtPoint(
    const cv::Mat& sem_map, const cv::Point& pt) 
{
  Eigen::VectorXf feat = Eigen::VectorXf::Zero(8);
  int cls = sem_map.at<uint8_t>(pt);
  if (cls < feat.size()) {
    feat[cls] = 1;
  }
  return feat;
}

void AerialMap::InferenceThread::fitModel() {
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
    features = Eigen::ArrayXXf(8, max_features);
    labels = Eigen::VectorXi(max_features);

    for(int i=0; i<aerial_map_.reachability_map_.map.rows; ++i) {
      for(int j=0; j<aerial_map_.reachability_map_.map.cols; ++j) {
        cv::Point pt(j, i);
        int16_t trav_map_pt = aerial_map_.reachability_map_.map.at<int16_t>(pt);
        if (trav_map_pt <= -aerial_map_.params_.not_trav_thresh) {
          features.col(n_features) = getFeatureAtPoint(
              aerial_map_.feature_map_.map, pt);
          labels[n_features] = 0;
          ++n_features;
        } else if (trav_map_pt >= aerial_map_.params_.trav_thresh) {
          features.col(n_features) = getFeatureAtPoint(
              aerial_map_.feature_map_.map, pt);
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

cv::Mat AerialMap::InferenceThread::updateProbabilityMap() {
  cv::Mat prob_map;
  if (!model_.trained()) return prob_map;

  Eigen::ArrayXXf features;
  int feat_cnt = 0;
  std::vector<float*> prob_ptrs;
  {
    std::scoped_lock lock(aerial_map_.feature_map_.mtx);
    int n_features = cv::countNonZero(aerial_map_.feature_map_.map < 255);
    if (n_features == 0) return prob_map;
        
    prob_map = cv::Mat::zeros(aerial_map_.feature_map_.map.size(), CV_32FC1);
    features = Eigen::ArrayXXf(8, n_features);
    prob_ptrs.resize(n_features);

    for(int i=0; i<prob_map.rows; ++i) {
      for(int j=0; j<prob_map.cols; ++j) {
        cv::Point pt(j, i);

        if (aerial_map_.feature_map_.map.at<uint8_t>(pt) < 255) {
          prob_ptrs[feat_cnt] = prob_map.ptr<float>(i, j);
          features.col(feat_cnt) = getFeatureAtPoint(
              aerial_map_.feature_map_.map, pt);
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
