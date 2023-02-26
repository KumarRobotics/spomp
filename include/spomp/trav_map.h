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
      bool learn_trav = false;
      bool uniform_node_sampling = false;
      bool no_max_terrain_in_graph = true;
      float max_hole_fill_size_m = 1;
      float vis_dist_m = 10;
      float unvis_start_thresh = 10;
      float unvis_stop_thresh = 2;
      bool prune = true;
      float max_prune_edge_dist_m = 20;
      float recover_reset_dist_m = 10;
      // Not set directly
      int num_robots = 1;
    };
    TravMap(const Params& tm_p, const TravGraph::Params& tg_p, 
        const AerialMapInfer::Params& am_p, const MLPModel::Params& mlp_p);

    void updateMap(const cv::Mat& map, const Eigen::Vector2f& center, 
        const std::vector<cv::Mat>& other_maps = {});
    std::list<TravGraph::Node*> getPath(const Eigen::Vector2f& start_p,
        const Eigen::Vector2f& end_p);
    std::list<TravGraph::Node*> getPath(TravGraph::Node& start_n,
        TravGraph::Node& end_n);
    float getPathCost(const std::list<TravGraph::Node*>& path) {
      return graph_.getPathCost(path);
    }
    //! @return True if graph changed
    // May want to also flag if aerial map changes
    bool updateLocalReachability(const Reachability& reachability, int robot_id = 0);
    bool updateEdgeFromReachability(TravGraph::Edge& edge, 
        const TravGraph::Node& start_node, const Reachability& reachability,
        std::optional<Eigen::Vector2f> start_pos = {}) {
      return graph_.updateEdgeFromReachability(edge, start_node, reachability, start_pos);
    }

    void resetGraphAroundPoint(const Eigen::Vector2f& pt);

    cv::Mat viz() const;
    cv::Mat viz_visibility() const;

    const auto& getEdges() const {
      return graph_.getEdges();
    }

    const auto& getMapReferenceFrame() const {
      return map_ref_frame_;
    }

    const cv::Mat getAerialMapTrav() {
      return aerial_map_->viz();
    }

    const auto& getReachabilityHistory() const {
      return reach_hist_[0];
    }

    bool haveReachabilityForRobotAtStamp(int robot_id, uint64_t stamp) const {
      if (robot_id < reach_hist_.size()) {
        return reach_hist_[robot_id].count(stamp) > 0;
      } else {
        return false;
      }
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
    void addEdge(const TravGraph::Edge& edge);
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
    
    TravGraph graph_;
    // This has to be a pointer because AerialMap is a virtual interface class
    std::unique_ptr<AerialMap> aerial_map_;

    std::vector<std::unordered_map<uint64_t, Reachability>> reach_hist_;

    Timer* compute_dist_maps_t_;
    Timer* reweight_graph_t_;
    Timer* rebuild_visibility_t_;
    Timer* build_graph_t_;
};

} // namespace spomp
