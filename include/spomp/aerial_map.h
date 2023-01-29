#pragma once

#include <thread>
#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"
#include "spomp/mlp_model.h"

namespace spomp {

class AerialMap {
  public:
    struct Params {
      // This is really short for test purposes
      int inference_thread_period_ms = 10;
      int trav_thresh = 1;
      int not_trav_thresh = 1;
      float not_trav_range_m = 3;
    };
    AerialMap(const Params& p, const MLPModel::Params& mlp_p);
    ~AerialMap();

    void updateMap(const cv::Mat& sem_map, const MapReferenceFrame& mrf);

    void updateLocalReachability(const Reachability& reach);

    float getEdgeProb(const Eigen::Vector2f& n1, const Eigen::Vector2f& n2) const;

    cv::Mat viz();

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    struct FeatureMap {
      std::mutex mtx;
      cv::Mat map;
    } feature_map_;

    MapReferenceFrame map_ref_frame_;

    struct ReachabilityMap {
      std::mutex mtx;
      cv::Mat map;
    } reachability_map_;

    struct ProbabilityMap {
      std::mutex mtx;
      cv::Mat map;
    } prob_map_;


    Timer* update_reachability_t_;

    /*********************************************************
     * THREADS
     *********************************************************/
    std::atomic<bool> exit_threads_flag_ = false;

    std::thread inference_thread_;
    class InferenceThread {
      public:
        InferenceThread(AerialMap& am, const MLPModel::Params& mlp_p) : 
          aerial_map_(am), model_(mlp_p) {}

        bool operator()();

        void fitModel();
        static Eigen::VectorXf getFeatureAtPoint(const cv::Mat& sem_map, 
            const cv::Point& pt);
        cv::Mat updateProbabilityMap();

      private:
        MLPModel model_;

        // Reference back to parent
        AerialMap& aerial_map_;

        Timer* model_fit_t_;
        Timer* update_probability_map_t_;
    };
};

} // namespace spomp
