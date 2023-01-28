#pragma once

#include <set>
#include <filesystem>
#include <opencv2/core.hpp>
#include "spomp/trav_graph.h"
#include "spomp/aerial_map.h"
#include "spomp/timer.h"
#include "semantics_manager/semantics_manager.h"

namespace spomp {

class TravMap {
  public:
    struct Params {
      std::string world_config_path = "";
      float max_hole_fill_size_m = 2;
      float vis_dist_m = 10;
      float unvis_start_thresh = 0.1;
      float unvis_stop_thresh = 0.01;
      bool prune = true;
    };
    TravMap(const Params& tm_p, const TravGraph::Params& tg_p, 
        const AerialMap::Params& am_p, const MLPModel::Params& mlp_p);

    void updateMap(const cv::Mat& map, const Eigen::Vector2f& center);
    std::list<TravGraph::Node*> getPath(const Eigen::Vector2f& start_p,
        const Eigen::Vector2f& end_p);
    std::list<TravGraph::Node*> getPath(TravGraph::Node& start_n,
        TravGraph::Node& end_n);
    float getPathCost(const std::list<TravGraph::Node*>& path) {
      return graph_.getPathCost(path);
    }
    //! @return True if graph changed
    // May want to also flag if aerial map changes
    bool updateLocalReachability(const Reachability& reachability) {
      aerial_map_.updateLocalReachability(reachability);
      if (reach_cnt % 5 == 0) {
        aerial_map_.fitModel();
      }
      ++reach_cnt;
      return graph_.updateLocalReachability(reachability);
    }
    bool updateEdgeFromReachability(TravGraph::Edge& edge, 
        const TravGraph::Node& start_node, const Reachability& reachability,
        std::optional<Eigen::Vector2f> start_pos = {}) {
      return graph_.updateEdgeFromReachability(edge, start_node, reachability, start_pos);
    }

    cv::Mat viz() const;
    cv::Mat viz_visibility() const;

    const auto& getEdges() const {
      return graph_.getEdges();
    }

    const auto& getMapReferenceFrame() const {
      return map_ref_frame_;
    }

    const cv::Mat getAerialMapTrav() const {
      return aerial_map_.viz();
    }

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    void loadClasses(const semantics_manager::ClassConfig& class_config);
    void loadStaticMap(const semantics_manager::MapConfig& map_config,
        const semantics_manager::ClassConfig& class_config);
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
    bool dynamic_{true};

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    cv::Mat map_{};
    MapReferenceFrame map_ref_frame_{};

    std::vector<cv::Mat> dist_maps_{};
    cv::Mat visibility_map_{};
    int reach_cnt = 0;
    
    TravGraph graph_;
    AerialMap aerial_map_;

    Timer* compute_dist_maps_t_;
    Timer* reweight_graph_t_;
    Timer* rebuild_visibility_t_;
    Timer* build_graph_t_;
};

} // namespace spomp
