#pragma once

#include <thread>
#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"
#include "spomp/mlp_model.h"

namespace spomp {

class AerialMap {
  public:
    virtual ~AerialMap() {}

    virtual void updateMap(const cv::Mat& sem_map, 
        const std::vector<cv::Mat>& dm, const MapReferenceFrame& mrf,
        const cv::Mat& color_map = cv::Mat()) {}

    struct EdgeInfo {
      int cls = 0;
      float cost = 0;
    };
    virtual EdgeInfo traceEdge(const Eigen::Vector2f& n1, 
        const Eigen::Vector2f& n2) {return {};}

    virtual void updateLocalReachability(const Reachability& reach) {}

    virtual cv::Mat viz() { 
      return cv::Mat::zeros(/*rows*/ map_ref_frame_.size[1], 
        /*cols*/ map_ref_frame_.size[0], CV_8UC3);
    }

    virtual bool haveNewTrav() { 
      // Want to default to false
      // This way we don't unneccessarily rebuild graph costs
      return false; 
    }

    virtual void setTravRead() {}

  protected:
    MapReferenceFrame map_ref_frame_{};
};

class AerialMapPrior : public AerialMap {
  public:
    AerialMapPrior() = default;

    void updateMap(const cv::Mat& sem_map, 
        const std::vector<cv::Mat>& dm, const MapReferenceFrame& mrf,
        const cv::Mat& color_map = cv::Mat());

    EdgeInfo traceEdge(const Eigen::Vector2f& n1, const Eigen::Vector2f& n2);

  private:
    cv::Mat map_{};
    std::vector<cv::Mat> dist_maps_{};
};

class AerialMapInfer : public AerialMap {
  public:
    struct Params {
      // This is really short for test purposes
      int inference_thread_period_ms = 10;
      int trav_thresh = 1;
      int not_trav_thresh = 1;
      float not_trav_range_m = 3;
    };
    AerialMapInfer(const Params& p, const MLPModel::Params& mlp_p, int n_cls);
    ~AerialMapInfer();

    void updateMap(const cv::Mat& sem_map, 
        const std::vector<cv::Mat>& dm, const MapReferenceFrame& mrf,
        const cv::Mat& color_map = cv::Mat());

    void updateLocalReachability(const Reachability& reach);

    EdgeInfo traceEdge(const Eigen::Vector2f& n1, const Eigen::Vector2f& n2);

    bool haveNewTrav() {
      std::scoped_lock lock(prob_map_.mtx);
      return prob_map_.have_new;
    }

    void setTravRead() {
      std::scoped_lock lock(prob_map_.mtx);
      prob_map_.have_new = false;
    }

    cv::Mat viz();

  private:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;
    static int feature_size_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    struct FeatureMap {
      std::mutex mtx;
      cv::Mat sem_map;
      cv::Mat color_map;
      std::vector<cv::Mat> dist_maps;
    } feature_map_;

    struct ReachabilityMap {
      std::mutex mtx;
      cv::Mat map;
    } reachability_map_;

    struct ProbabilityMap {
      std::mutex mtx;
      cv::Mat map;
      bool have_new{false};
    } prob_map_;


    Timer* update_reachability_t_;

    /*********************************************************
     * THREADS
     *********************************************************/
    std::atomic<bool> exit_threads_flag_ = false;

    std::thread inference_thread_;
    class InferenceThread {
      public:
        InferenceThread(AerialMapInfer& am, const MLPModel::Params& mlp_p) : 
          aerial_map_(am), model_(mlp_p) {}

        bool operator()();

        void fitModel();
        static Eigen::VectorXf getFeatureAtPoint(const FeatureMap& feat_map, 
            const cv::Point& pt);
        cv::Mat updateProbabilityMap();

      private:
        MLPModel model_;

        // Reference back to parent
        AerialMapInfer& aerial_map_;

        Timer* model_fit_t_;
        Timer* update_probability_map_t_;
    };
};

} // namespace spomp
