#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include "spomp/trav_map.h"
#include "spomp/utils.h"

namespace spomp {

TravMap::TravMap(const Params& tm_p, const TravGraph::Params& tg_p, 
    const AerialMapInfer::Params& am_p, const MLPModel::Params& mlp_p) : 
  params_(tm_p), graph_(tg_p)
{
  auto& tm = TimerManager::getGlobal(true);
  compute_dist_maps_t_ = tm.get("TM_compute_dist_maps");
  reweight_graph_t_ = tm.get("TM_reweight_graph");
  rebuild_visibility_t_ = tm.get("TM_rebuild_visibility");
  build_graph_t_ = tm.get("TM_build_graph");

  reach_hist_.resize(tm_p.num_robots);

  terrain_lut_ = cv::Mat::ones(256, 1, CV_8UC1) * 255;
  if (params_.world_config_path != "") {
    semantics_manager::ClassConfig class_config(
        semantics_manager::getClassesPath(params_.world_config_path));
    semantics_manager::MapConfig map_config(
        semantics_manager::getMapPath(params_.world_config_path));
    loadClasses(class_config);

    if (params_.learn_trav) {
      aerial_map_ = std::make_unique<AerialMapInfer>(am_p, mlp_p,
          TravGraph::Edge::MAX_TERRAIN);
    } else {
      aerial_map_ = std::make_unique<AerialMapPrior>();
    }
    loadStaticMap(map_config, class_config);
  } else {
    std::cout << "\033[31m" << "[ERROR] Using TravMap defaults.  Really a test-only case." 
      << "\033[0m" << std::endl;
    terrain_lut_.at<uint8_t>(0) = 0;
    terrain_lut_.at<uint8_t>(1) = 1;
    map_ref_frame_.res = 1;

    if (params_.learn_trav) {
      aerial_map_ = std::make_unique<AerialMapInfer>(am_p, mlp_p, 2);
    } else {
      aerial_map_ = std::make_unique<AerialMapPrior>();
    }
  }
}

void TravMap::loadClasses(const semantics_manager::ClassConfig& class_config) {
  int terrain_ind = 0;
  for (const auto& terrain_type : class_config.traversabililty_diff) {
    terrain_lut_.at<uint8_t>(terrain_ind) = terrain_type;
    int terrain_type_tmp = terrain_type;
    if (!params_.no_max_terrain_in_graph) {
      terrain_type_tmp += 1;
    }
    if (terrain_type_tmp > TravGraph::Edge::MAX_TERRAIN) {
      TravGraph::Edge::MAX_TERRAIN = terrain_type_tmp;
    }
    ++terrain_ind;
  }
}

void TravMap::loadStaticMap(const semantics_manager::MapConfig& map_config, 
    const semantics_manager::ClassConfig& class_config) 
{
  map_ref_frame_.res = map_config.resolution;

  // Initialize dist_maps_
  dist_maps_.reserve(TravGraph::Edge::MAX_TERRAIN);
  for (int terrain_ind=0; terrain_ind<TravGraph::Edge::MAX_TERRAIN; ++terrain_ind) {
    dist_maps_.emplace_back();
  }

  if (map_config.dynamic) {
    dynamic_ = true;
    std::cout << "\033[36m" << "[SPOMP-Global] Using dynamic map" << "\033[0m" << std::endl;
    return;
  } else {
    std::cout << "\033[36m" << "[SPOMP-Global] Loading static map..." << "\033[0m" << std::endl;
  }

  cv::Mat color_sem, class_sem, color_map;
  color_sem = cv::imread(map_config.raster_path);
  if (color_sem.data == NULL) {
    std::cout << "\033[31m" << "[ERROR] Static Map path specified but not read correctly" 
      << "\033[0m" << std::endl;
    return;
  }

  if (!map_config.color_path.empty()) {
    color_map = cv::imread(map_config.color_path);
    if (color_map.data == NULL) {
      std::cout << "\033[31m" << "[ERROR] Color Map path specified but not read correctly" 
        << "\033[0m" << std::endl;
    } else {
      cv::rotate(color_map, color_map, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
  }

  class_config.color_lut.color2Ind(color_sem, class_sem);
  Eigen::Vector2f map_center = Eigen::Vector2f(class_sem.cols, class_sem.rows)/2 / 
    map_config.resolution;

  // Set map now so that world2img works properly
  cv::rotate(class_sem, class_sem, cv::ROTATE_90_COUNTERCLOCKWISE);

  // We have set map_center_ already, but want to postproc
  updateMap(class_sem, map_center, {color_map});
  // Set this to block updateMap from doing anything in the future
  dynamic_ = false;
  std::cout << "\033[36m" << "[SPOMP-Global] Static Map loaded" << "\033[0m" << std::endl;
}

void TravMap::updateMap(const cv::Mat &map, const Eigen::Vector2f& center, 
    const std::vector<cv::Mat>& other_maps)
{
  if (!dynamic_) return;

  if (map.channels() == 1) {
    map_ = map;
  } else {
    // In the event we receive a 3 channel image, assume all channels are
    // the same
    cv::cvtColor(map, map_, cv::COLOR_BGR2GRAY);
  }
  cv::LUT(map_, terrain_lut_, map_);
  map_ref_frame_.center = center;
  map_ref_frame_.setMapSizeFrom(map_);

  computeDistMaps();
  aerial_map_->updateMap(map_, dist_maps_, map_ref_frame_, other_maps);

  reweightGraph();
  rebuildVisibility();
  buildGraph();
}

std::list<TravGraph::Node*> TravMap::getPath(const Eigen::Vector2f& start_p,
    const Eigen::Vector2f& end_p)
{
  if (map_.empty()) {
    return {};
  }

  auto n1_img_pos = map_ref_frame_.world2img(start_p);
  auto n2_img_pos = map_ref_frame_.world2img(end_p);
  if (!map_ref_frame_.imgPointInMap(n1_img_pos) || 
      !map_ref_frame_.imgPointInMap(n2_img_pos))
  {
    // Out of bounds
    return {};
  }

  int n1_id = visibility_map_.at<int32_t>(cv::Point(n1_img_pos[0], n1_img_pos[1]));
  int n2_id = visibility_map_.at<int32_t>(cv::Point(n2_img_pos[0], n2_img_pos[1]));
  if (n1_id < 0 || n2_id < 0) {
    // At least one of the endpoints not visible
    return {};
  }

  TravGraph::Node *n1 = &graph_.getNode(n1_id);
  TravGraph::Node *n2 = &graph_.getNode(n2_id);

  // Add start and end nodes to graph, if they are far enough away
  if ((start_p - n1->pos).norm() > 2) {
    auto worst_edge = aerial_map_->traceEdge(start_p, n1->pos);
    auto img_loc = map_ref_frame_.world2img(start_p);
    int start_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
    auto new_n = addNode(start_p, start_cls);
    addEdge({n1, new_n, worst_edge.cost, worst_edge.cls});
    n1 = new_n;
  }
  if ((end_p - n2->pos).norm() > 2) {
    auto worst_edge = aerial_map_->traceEdge(end_p, n2->pos);
    auto img_loc = map_ref_frame_.world2img(end_p);
    int end_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
    auto new_n = addNode(end_p, end_cls);
    addEdge({n2, new_n, worst_edge.cost, worst_edge.cls});
    n2 = new_n;
  }

  return getPath(*n1, *n2);
}

bool TravMap::updateLocalReachability(const Reachability& reachability, int robot_id) {
  if (robot_id < reach_hist_.size()) {
    if (reach_hist_[robot_id].size() == 0 || 
        (reach_hist_[robot_id].cbegin()->second.getPose().translation() - 
         reachability.getPose().translation()).norm() > 2) 
    {
      reach_hist_[robot_id].insert({reachability.getStamp(), reachability});
    }
  }
  aerial_map_->updateLocalReachability(reachability);
  if (aerial_map_->haveNewTrav()) {
    reweightGraph();
    aerial_map_->setTravRead();
  }
  return graph_.updateLocalReachability(reachability);
}

std::list<TravGraph::Node*> TravMap::getPath(TravGraph::Node& start_n, 
    TravGraph::Node& end_n) 
{
  auto path = graph_.getPath(&start_n, &end_n);
  auto final_path = path;
  if (final_path.size() < 1) {
    return final_path;
  }

  float original_cost = path.back()->cost;
  if (original_cost >= std::pow(1000, TravGraph::Edge::MAX_TERRAIN)-1) 
  {
    // If cost is this high, we have an obstacle edge
    path = {};
  }

  if (params_.prune) {
    final_path = prunePath(path);
    if (getPathCost(final_path) > original_cost) {
      // This case can happen because the pruning logic does not
      // consider experienced reachability
      final_path = path;
    }
  }
  return final_path;
}

std::list<TravGraph::Node*> TravMap::prunePath(
    const std::list<TravGraph::Node*>& path) 
{
  std::list<TravGraph::Node*> pruned_path;
  if (path.size() < 1) return pruned_path;

  TravGraph::Node* last_node = path.front();
  TravGraph::Edge last_edge;
  float summed_cost = 0;
  for (const auto& node : path) {
    if (pruned_path.size() == 0) {
      pruned_path.push_back(node);
    } else {
      summed_cost += node->getEdgeToNode(last_node)->totalCost();

      const auto edge_info = aerial_map_->traceEdge(pruned_path.back()->pos, node->pos);
      TravGraph::Edge direct_edge(pruned_path.back(), node, edge_info.cost, edge_info.cls);

      // Add extra check that the two edges are not the same
      // Costs can vary slightly because they can vary depending on which direction
      // they are computed
      if ((direct_edge.length > params_.max_prune_edge_dist_m ||
           direct_edge.totalCost() > summed_cost + 0.01) &&
          pruned_path.back() != last_node) 
      {
        // The direct cost is now the single last leg
        summed_cost = node->getEdgeToNode(last_node)->totalCost();

        if (!(last_node->getEdgeToNode(pruned_path.back())) &&
            last_edge.node1 && last_edge.node2) 
        {
          // No edge found, so add
          addEdge(last_edge);
          last_edge = TravGraph::Edge();
        }
        pruned_path.push_back(last_node);
      }
      last_node = node;
      last_edge = direct_edge;
    }
  }

  // Do check again for last pt
  if (!(path.back()->getEdgeToNode(pruned_path.back())) &&
      last_edge.node1 && last_edge.node2) 
  {
    // No edge found, so add
    addEdge(last_edge);
  }
  // Update the node costs
  pruned_path.push_back(path.back());

  return pruned_path;
}

void TravMap::computeDistMaps() {
  compute_dist_maps_t_->start();

  auto morph_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(
        params_.max_hole_fill_size_m * map_ref_frame_.res * 2 + 1,
        params_.max_hole_fill_size_m * map_ref_frame_.res * 2 + 1));

  int t_cls = 0;
  for (auto& dist_map : dist_maps_) {
    cv::Mat trav_map;
    // 1 everywhere where we want to compute distance
    cv::bitwise_or(map_ <= t_cls, map_ == 255, trav_map);
    cv::morphologyEx(trav_map, trav_map, cv::MORPH_CLOSE, morph_kernel);
    cv::distanceTransform(trav_map, dist_map, cv::DIST_L2, cv::DIST_MASK_5);
    dist_map.setTo(0, map_ == 255);
    ++t_cls;
  }

  compute_dist_maps_t_->end();
}

void TravMap::rebuildVisibility() {
  rebuild_visibility_t_->start();

  // This is brute-force, can definitely do better
  visibility_map_ = -cv::Mat::ones(map_.rows, map_.cols, CV_32SC1);
  for (const auto& [node_id, node] : graph_.getNodes()) {
    // Rebuild node visibility
    addNodeToVisibility(node);
  }

  rebuild_visibility_t_->end();
}

void TravMap::reweightGraph() {
  reweight_graph_t_->start();

  for (auto& edge : graph_.getEdges()) {
    // Only reweight if we don't already have first-hand experience
    if (!edge.is_locked && !edge.is_experienced) {
      auto edge_info = aerial_map_->traceEdge(edge.node1->pos, edge.node2->pos);
      edge.cls = edge_info.cls;
      edge.cost = edge_info.cost;
    }
  }

  for (auto& [node_id, node] : graph_.getNodes()) {
    auto img_loc = map_ref_frame_.world2img(node.pos);
    int node_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
    if (node_cls == TravGraph::Edge::MAX_TERRAIN) {
      // Node is no longer traversable, sad
      visibility_map_.setTo(-1, visibility_map_ == node_id);
    }
  }

  reweight_graph_t_->end();
}

void TravMap::buildGraph() {
  build_graph_t_->start();

  double min_v, max_v;
  cv::Point min_l, max_l;

  auto morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(
        params_.max_hole_fill_size_m * map_ref_frame_.res * 2 + 1,
        params_.max_hole_fill_size_m * map_ref_frame_.res * 2 + 1));

  int start_cls = 0;
  if (params_.uniform_node_sampling) {
    // Uniform sampling: just start at last class, which effectively
    // ignores all other classes
    start_cls = TravGraph::Edge::MAX_TERRAIN-1;
  }

  for (int t_cls=start_cls; t_cls<TravGraph::Edge::MAX_TERRAIN; ++t_cls) {
    int num_to_cover = cv::countNonZero(map_ <= t_cls);
    cv::Mat dist_map_masked = dist_maps_[t_cls].clone();
    dist_map_masked.setTo(0, visibility_map_ >= 0);

    cv::morphologyEx(dist_map_masked, dist_map_masked, 
        cv::MORPH_OPEN, morph_kernel);

    if (cv::countNonZero(dist_map_masked) < 
        params_.unvis_start_thresh * std::pow(map_ref_frame_.res, 2)) continue;

    while (cv::countNonZero(dist_map_masked) > 
           params_.unvis_stop_thresh * std::pow(map_ref_frame_.res, 2)) 
    {
      // Ignore points already visible
      dist_map_masked.setTo(0, visibility_map_ >= 0);
      // Select next best node location
      cv::minMaxLoc(dist_map_masked, &min_v, &max_v, &min_l, &max_l);
      if (max_v == 0) {
        // If best point is 0, then clearly we are done here
        break;
      }

      auto n_ptr = addNode(map_ref_frame_.img2world({max_l.x, max_l.y}), t_cls);
      // Make absolutely sure that we don't pick the same pt again
      visibility_map_.at<int32_t>(max_l) = n_ptr->id;
    }
  }

  build_graph_t_->end();
}

TravGraph::Node* TravMap::addNode(const Eigen::Vector2f& pos, int t_cls) {
  TravGraph::Node* n = graph_.addNode(pos);
  auto overlapping_nodes = addNodeToVisibility(*n);
  //std::cout << "=============" << std::endl;
  //std::cout << "target cls: " << t_cls << std::endl;
  //std::cout << n->pos.transpose() << std::endl;
  //std::cout << max_v << std::endl;
  //std::cout << max_l << std::endl;
  //std::cout << cv::countNonZero(visibility_map_ >= 0) << std::endl;
  //std::cout << num_to_cover << std::endl;

  for (const auto& [overlap_n_id, overlap_loc] : overlapping_nodes) {
    auto overlap_n = &graph_.getNode(overlap_n_id);
    auto worst_edge = aerial_map_->traceEdge(n->pos, overlap_n->pos);
    if (worst_edge.cls <= t_cls) {
      addEdge({n, overlap_n, worst_edge.cost, worst_edge.cls});
    } else if (map_.at<uint8_t>(cv::Point(overlap_loc[0], overlap_loc[1])) <= t_cls) {
      // Add new intermediate node
      auto intermed_n = graph_.addNode({map_ref_frame_.img2world(overlap_loc)});
      addNodeToVisibility(*intermed_n);
      auto edge_info = aerial_map_->traceEdge(n->pos, intermed_n->pos);
      addEdge({n, intermed_n, edge_info.cost, edge_info.cls});
      edge_info = aerial_map_->traceEdge(overlap_n->pos, intermed_n->pos);
      addEdge({overlap_n, intermed_n, edge_info.cost, edge_info.cls});
    }
  }

  return n;
}

void TravMap::addEdge(const TravGraph::Edge& edge) {
  TravGraph::Edge* inserted_edge = graph_.addEdge(edge);

  if (inserted_edge) {
    for (const auto& robot_reach_hist : reach_hist_) {
      for (const auto& [stamp, reach] : robot_reach_hist) {
        updateEdgeFromReachability(*inserted_edge, *edge.node1, reach);
      }
    }
  }
}

std::map<int, Eigen::Vector2f> TravMap::addNodeToVisibility(const TravGraph::Node& n) {
  // Get class of node location
  auto img_loc = map_ref_frame_.world2img(n.pos);
  int node_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
  //std::cout << "+++++++++++" << std::endl;
  //std::cout << img_loc.transpose() << std::endl;
  //std::cout << node_cls << std::endl;

  std::map<int, Eigen::Vector2f> overlapping_nodes;

  // Choose delta such that ends of rays are within 0.5 cells
  float delta_t = 0.5 / (params_.vis_dist_m * map_ref_frame_.res);
  for (float theta=0; theta<2*pi; theta+=delta_t) {
    Eigen::Vector2f dir = {cos(theta), sin(theta)};

    for (float r=0; r<map_ref_frame_.res*params_.vis_dist_m; r+=0.5) {
      Eigen::Vector2f img_cell = img_loc + dir*r;
      if ((img_cell.array() < 0).any() || 
          (img_cell.array() >= Eigen::Vector2f(map_.cols, map_.rows).array()).any()) {
        // We have left the image
        continue;
      }

      int cell_cls = map_.at<uint8_t>(cv::Point(img_cell[0], img_cell[1]));
      auto& vis_cell = visibility_map_.at<int32_t>(cv::Point(img_cell[0], img_cell[1]));

      // Always do this before breaking loop so we include endpoints
      if (vis_cell >= 0 && vis_cell != n.id) {
        overlapping_nodes.emplace(vis_cell, img_cell);
      }
      vis_cell = n.id;

      // If uniform sampling, ignore raytraced classes
      if (cell_cls < node_cls && cell_cls != 255 && !params_.uniform_node_sampling) {
        break;
      }
    }
  }

  return overlapping_nodes;
}

void TravMap::resetGraphAroundPoint(const Eigen::Vector2f& pt) {
  auto nodes = graph_.getNodesNear(pt, params_.recover_reset_dist_m); 

  for (auto& node : nodes) {
    for (auto& edge : node->edges) {
      // Only reweight if we don't already have first-hand experience
      auto edge_info = aerial_map_->traceEdge(edge->node1->pos, edge->node2->pos);
      edge->cls = edge_info.cls;
      edge->cost = edge_info.cost;
      edge->is_locked = false;
      edge->untrav_counter = 0;
    }
  }
}

void TravMap::resetGraphLocked() {
  for (auto& edge : graph_.getEdges()) {
    if (edge.cls > 0) {
      edge.is_locked = false;
    }
  }
}

cv::Mat TravMap::viz() const {
  cv::Mat viz;
  if (map_.empty()) {
    return viz;
  }

  cv::Mat scaled_map = map_.clone();
  // Rescale for increasing order of trav difficulty 0->255
  scaled_map *= 255. / TravGraph::Edge::MAX_TERRAIN;
  cv::Mat cmapped_map;
  cv::applyColorMap(scaled_map, cmapped_map, cv::COLORMAP_PARULA);
  // Mask unknown regions
  cmapped_map.copyTo(viz, map_ < 255);

  // Draw nodes
  for (const auto& [node_id, node] : graph_.getNodes()) {
    auto node_img_pos = map_ref_frame_.world2img(node.pos);
    cv::circle(viz, cv::Point(node_img_pos[0], node_img_pos[1]), 1, cv::Scalar(0, 255, 0), 2);
  }

  // Draw edges
  for (const auto& edge : graph_.getEdges()) {
    auto node1_img_pos = map_ref_frame_.world2img(edge.node1->pos);
    auto node2_img_pos = map_ref_frame_.world2img(edge.node2->pos);
    cv::Scalar color;
    if (edge.is_locked) {
      if (edge.cls == 0) {
        color = {0, 255, 0};
      } else {
        color = {0, 0, 255};
      }
    } else {
      float color_mag = std::min<float>(1./(edge.cost*edge.length), 1);
      float hue = (static_cast<float>(edge.cls)/
          (TravGraph::Edge::MAX_TERRAIN-1))*(1./2) + 1./2;
      Eigen::Vector3f rgb = hsv2rgb({hue, 0.5, color_mag});
      color = {rgb[2]*255, rgb[1]*255, rgb[0]*255};
    }
    cv::line(viz, cv::Point(node1_img_pos[0], node1_img_pos[1]),
             cv::Point(node2_img_pos[0], node2_img_pos[1]), color);
  }

  // Draw origin
  auto origin_img = map_ref_frame_.world2img({0, 0});
  cv::circle(viz, cv::Point(origin_img[0], origin_img[1]), 3, cv::Scalar(0, 0, 255), 2);
  
  return viz;
}

cv::Mat TravMap::viz_visibility() const {
  double min_v, max_v;
  cv::Point min_l, max_l;
  cv::minMaxLoc(visibility_map_, &min_v, &max_v, &min_l, &max_l);

  cv::Mat visibility_viz;
  visibility_map_.convertTo(visibility_viz, CV_8UC1, 255./graph_.size());
  cv::Mat cmapped_map;
  cv::applyColorMap(visibility_viz, cmapped_map, cv::COLORMAP_PARULA);
  return cmapped_map;
}

} // namespace spomp
