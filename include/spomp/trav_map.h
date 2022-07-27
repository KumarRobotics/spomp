#pragma once

#include <opencv2/core.hpp>
#include <set>
#include "spomp/trav_graph.h"
#include "spomp/timer.h"
#include "spomp/semantic_color_lut.h"
#include "spomp/utils.h"

namespace spomp {

class TravMap {
  public:
    struct Params {
      std::string terrain_types_path = "";
      std::string static_map_path = "";
      std::string semantic_lut_path = "";
      float map_res = 2;
      float max_hole_fill_size_m = 2;
      float vis_dist_m = 10;
      float unvis_start_thresh = 0.1;
      float unvis_stop_thresh = 0.01;
      bool prune = true;
    };
    TravMap(const Params& p);

    void updateMap(const cv::Mat& map, const Eigen::Vector2f& center);
    void updateLocalReachability(const Reachability& reachability, 
        const Eigen::Isometry3f& reach_pose) 
    {
    }
    std::list<TravGraph::Node*> getPath(const Eigen::Vector2f& start_p,
        const Eigen::Vector2f& end_p);

    Eigen::Vector2f world2img(const Eigen::Vector2f& world_c) const;
    Eigen::Vector2f img2world(const Eigen::Vector2f& img_c) const;

    cv::Mat viz() const;
    cv::Mat viz_visibility() const;

    const auto& getEdges() const {
      return graph_.getEdges();
    }

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void loadTerrainLUT();
    void loadStaticMap();
    void computeDistMaps();
    void rebuildVisibility();
    void reweightGraph();
    void buildGraph();
    std::list<TravGraph::Node*> prunePath(const std::list<TravGraph::Node*>& path);

    std::pair<int, float> traceEdge(const Eigen::Vector2f& n1, 
        const Eigen::Vector2f& n2);
    /*!
     * @param pos Position of node in world space
     * @param t_cls require that intermediate edges be this class or better
     */
    TravGraph::Node* addNode(const Eigen::Vector2f& pos, int t_cls);
    //! @return set of neighboring nodes
    std::map<int, Eigen::Vector2f> addNodeToVisibility(const TravGraph::Node& n);

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;
    cv::Mat terrain_lut_{};
    int max_terrain_{1};

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    cv::Mat map_{};
    Eigen::Vector2f map_center_{};

    std::vector<cv::Mat> dist_maps_{};
    cv::Mat visibility_map_{};
    
    TravGraph graph_{};

    Timer* compute_dist_maps_t_;
    Timer* reweight_graph_t_;
    Timer* rebuild_visibility_t_;
    Timer* build_graph_t_;
};

} // namespace spomp
