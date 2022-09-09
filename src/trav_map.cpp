#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include "spomp/trav_map.h"
#include "spomp/utils.h"

namespace spomp {

TravMap::TravMap(const Params& p) : params_(p) {
  auto& tm = TimerManager::getGlobal();
  compute_dist_maps_t_ = tm.get("TM_compute_dist_maps");
  reweight_graph_t_ = tm.get("TM_reweight_graph");
  rebuild_visibility_t_ = tm.get("TM_rebuild_visibility");
  build_graph_t_ = tm.get("TM_build_graph");

  loadTerrainLUT();
  loadStaticMap();
}

void TravMap::loadTerrainLUT() {
  terrain_lut_ = cv::Mat::ones(256, 1, CV_8UC1) * 255;
  if (params_.terrain_types_path != "") {
    const YAML::Node terrain_types = YAML::LoadFile(params_.terrain_types_path);
    int terrain_ind = 0;
    for (const auto& terrain_type : terrain_types) {
      int type = terrain_type.as<int>();
      terrain_lut_.at<uint8_t>(terrain_ind) = type;
      if (type > max_terrain_) {
        max_terrain_ = type;
      }
      ++terrain_ind;
    }
  } else {
    // Default: assume first class is traversable, not second
    // Rest unknown
    terrain_lut_.at<uint8_t>(0) = 0;
    terrain_lut_.at<uint8_t>(1) = 1;
  }

  // Initialize dist_maps_
  dist_maps_.reserve(max_terrain_);
  for (int terrain_ind=0; terrain_ind<max_terrain_; ++terrain_ind) {
    dist_maps_.emplace_back();
  }
}

void TravMap::loadStaticMap() {
  if (params_.static_map_path == "") {
    return;
  }

  SemanticColorLut semantic_color_lut;
  try {
    semantic_color_lut = SemanticColorLut(params_.semantic_lut_path);
  } catch (const std::exception& ex) {
    // This usually happens when there is a yaml reading error
    std::cout << "\033[31m" << "[ERROR] Cannot create semantic LUT: " << ex.what() 
      << "\033[0m" << std::endl;
    return;
  }

  cv::Mat color_sem, class_sem;
  color_sem = cv::imread(params_.static_map_path);
  if (color_sem.data == NULL) {
    std::cout << "\033[31m" << "[ERROR] Static Map path specified but not read correctly" 
      << "\033[0m" << std::endl;
    return;
  }

  semantic_color_lut.color2Ind(color_sem, class_sem);
  map_center_ = Eigen::Vector2f(class_sem.cols, class_sem.rows)/2 / params_.map_res;

  // Set map now so that world2img works properly
  cv::rotate(class_sem, class_sem, cv::ROTATE_90_COUNTERCLOCKWISE);

  // We have set map_center_ already, but want to postproc
  updateMap(class_sem, map_center_);
  std::cout << "\033[34m" << "[SPOMP-Global] Static Map loaded" << "\033[0m" << std::endl;
}

Eigen::Vector2f TravMap::world2img(const Eigen::Vector2f& world_c) const {
  Eigen::Vector2f world_pt = world_c - map_center_;
  Eigen::Vector2f img_pt = {-world_pt[1], -world_pt[0]};
  img_pt *= params_.map_res;
  img_pt += Eigen::Vector2f(map_.cols, map_.rows)/2;
  return img_pt;
}

Eigen::Vector2f TravMap::img2world(const Eigen::Vector2f& img_c) const {
  Eigen::Vector2f img_pt = img_c - Eigen::Vector2f(map_.cols, map_.rows)/2;
  img_pt /= params_.map_res;
  Eigen::Vector2f world_pt = {-img_pt[1], -img_pt[0]};
  world_pt += map_center_;
  return world_pt;
}

void TravMap::updateMap(const cv::Mat &map, const Eigen::Vector2f& center) {
  if (map.channels() == 1) {
    map_ = map;
  } else {
    // In the event we receive a 3 channel image, assume all channels are
    // the same
    cv::cvtColor(map, map_, cv::COLOR_BGR2GRAY);
  }
  cv::LUT(map_, terrain_lut_, map_);
  map_center_ = center;

  computeDistMaps();
  reweightGraph();
  rebuildVisibility();
  buildGraph();
}

void TravMap::updateLocalReachability(const Reachability& reachability, 
    const Eigen::Isometry2f& reach_pose)
{
  auto near_nodes = graph_.getNodesNear(reach_pose.translation(), 
      params_.reach_node_max_dist);

  for (const auto& node_ptr : near_nodes) {
    for (const auto& edge : node_ptr->edges) {
      TravGraph::Node* dest_node_ptr = edge->getOtherNode(node_ptr);
      Eigen::Vector2f local_dest_pose = reach_pose.inverse() * dest_node_ptr->pos;

      float range = local_dest_pose.norm();
      float bearing = atan2(local_dest_pose[1], local_dest_pose[0]);

      bool not_reachable = true;
      bool reachable = true;
      for (float b=bearing-0.1; b<=bearing+0.1; b+=std::abs(reachability.proj.delta_angle)) {
        int ind = reachability.proj.indAt(b);
        if (range <= reachability.scan[ind] || !reachability.is_obs[ind]) {
          // We have a non-obstacle path
          not_reachable = false;
        }
        if (range > reachability.scan[ind] && reachability.is_obs[ind]) {
          // We have an obstacle path
          reachable = false;
        }
      }

      if (not_reachable) {
        // Unreachable cost
        edge->cls = max_terrain_ + 1;
        edge->is_experienced = true;
      }
      if (reachable) {
        edge->cls = 0;
        edge->is_experienced = true;
      }
    }
  }
}

std::list<TravGraph::Node*> TravMap::getPath(const Eigen::Vector2f& start_p,
    const Eigen::Vector2f& end_p)
{
  std::list<TravGraph::Node*> path;

  auto n1_img_pos = world2img(start_p);
  auto n2_img_pos = world2img(end_p);
  if ((n1_img_pos.array() < 0).any() || 
      (n1_img_pos.array() >= Eigen::Vector2f(map_.cols, map_.rows).array()).any() ||
      (n2_img_pos.array() < 0).any() ||
      (n2_img_pos.array() >= Eigen::Vector2f(map_.cols, map_.rows).array()).any())
  {
    // Out of bounds
    return path;
  }

  int n1_id = visibility_map_.at<int32_t>(cv::Point(n1_img_pos[0], n1_img_pos[1]));
  int n2_id = visibility_map_.at<int32_t>(cv::Point(n2_img_pos[0], n2_img_pos[1]));
  if (n1_id < 0 || n2_id < 0) {
    // At least one of the endpoints not visible
    return path;
  }

  TravGraph::Node *n1 = &graph_.getNode(n1_id);
  TravGraph::Node *n2 = &graph_.getNode(n2_id);

  path = graph_.getPath(n1, n2);
  if (path.size() > 0) {
    if (path.back()->cost >= std::pow(1000, max_terrain_)-1) {
      // If cost is this high, we have an obstacle edge
      path = {};
    } else {
      // Add start and end nodes to graph, if they are far enough away
      if ((start_p - path.front()->pos).norm() > 2) {
        auto [worst_cls, worst_cost] = traceEdge(start_p, path.front()->pos);
        auto new_n = addNode(start_p, -1);
        graph_.addEdge({path.front(), new_n, worst_cost, worst_cls});
        path.push_front(new_n);
      }
      if ((end_p - path.back()->pos).norm() > 2) {
        auto [worst_cls, worst_cost] = traceEdge(end_p, path.back()->pos);
        auto new_n = addNode(end_p, -1);
        graph_.addEdge({path.back(), new_n, worst_cost, worst_cls});
        path.push_back(new_n);
      }
    }
  }

  auto final_path = path;
  if (params_.prune) {
    final_path = prunePath(path);
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
  float total_path_cost = 0;
  for (const auto& node : path) {
    if (pruned_path.size() == 0) {
      pruned_path.push_back(node);
    } else {
      summed_cost += node->getEdgeToNode(last_node)->totalCost();

      const auto [edge_cls, edge_cost] = traceEdge(pruned_path.back()->pos, node->pos);
      TravGraph::Edge direct_edge(pruned_path.back(), node, edge_cost, edge_cls);
      if (direct_edge.totalCost() > summed_cost + 0.01) {
        // The direct cost is now the single last leg
        summed_cost = node->getEdgeToNode(last_node)->totalCost();

        if (!(last_node->getEdgeToNode(pruned_path.back())) &&
            last_edge.node1 && last_edge.node2) 
        {
          // No edge found, so add
          graph_.addEdge(last_edge);
          last_edge = TravGraph::Edge();
        }
        total_path_cost += last_node->getEdgeToNode(pruned_path.back())->totalCost();
        // Update the node costs
        last_node->cost = total_path_cost;
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
    graph_.addEdge(last_edge);
  }
  total_path_cost += path.back()->getEdgeToNode(pruned_path.back())->totalCost();
  // Update the node costs
  last_node->cost = total_path_cost;
  pruned_path.push_back(path.back());

  return pruned_path;
}

void TravMap::computeDistMaps() {
  compute_dist_maps_t_->start();

  int t_cls = 0;
  auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(
        params_.max_hole_fill_size_m * params_.map_res,
        params_.max_hole_fill_size_m * params_.map_res));

  for (auto& dist_map : dist_maps_) {
    cv::Mat trav_map;
    cv::bitwise_or(map_ <= t_cls, map_ == 255, trav_map);
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
    if (!edge.is_experienced) {
      auto [edge_cls, edge_cost] = traceEdge(edge.node1->pos, edge.node2->pos);
      edge.cls = edge_cls;
      edge.cost = edge_cost;
    }
  }

  for (auto& [node_id, node] : graph_.getNodes()) {
    auto img_loc = world2img(node.pos);
    int node_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
    if (node_cls == max_terrain_) {
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

  for (int t_cls=0; t_cls<max_terrain_; ++t_cls) {
    int num_to_cover = cv::countNonZero(map_ <= t_cls);
    cv::Mat dist_map_masked = dist_maps_[t_cls].clone();
    dist_map_masked.setTo(0, visibility_map_ >= 0);

    if (cv::countNonZero(dist_map_masked) < 
        params_.unvis_start_thresh * num_to_cover) continue;

    while (cv::countNonZero(dist_map_masked) > 
           params_.unvis_stop_thresh * num_to_cover) 
    {
      // Ignore points already visible
      dist_map_masked.setTo(0, visibility_map_ >= 0);
      // Select next best node location
      cv::minMaxLoc(dist_map_masked, &min_v, &max_v, &min_l, &max_l);
      if (max_v == 0) {
        // If best point is 0, then clearly we are done here
        break;
      }

      auto n_ptr = addNode(img2world({max_l.x, max_l.y}), t_cls);
      // Make absolutely sure that we don't pick the same pt again
      visibility_map_.at<int32_t>(max_l) = n_ptr->id;
    }
  }

  build_graph_t_->end();
}

std::pair<int, float> TravMap::traceEdge(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2)
{
  auto img_pt1 = world2img(n1);
  auto img_pt2 = world2img(n2);
  float dist = (img_pt1 - img_pt2).norm();
  Eigen::Vector2f dir = (img_pt2 - img_pt1).normalized();

  int worst_cls = 0;
  float worst_dist = std::numeric_limits<float>::infinity();

  for (float cur_dist=0; cur_dist<dist; cur_dist+=0.5) {
    Eigen::Vector2f sample_pt = img_pt1 + dir*cur_dist;
    auto t_cls = map_.at<uint8_t>(cv::Point(sample_pt[0], sample_pt[1]));
    float dist = 0;
    if (t_cls < max_terrain_) {
      dist = dist_maps_[t_cls].at<float>(cv::Point(sample_pt[0], sample_pt[1]));
      //std::cout << sample_pt.transpose() << std::endl;
      //std::cout << dist << std::endl;
    
      if (t_cls > worst_cls) {
        worst_cls = t_cls;
        worst_dist = dist;
      } else if (dist < worst_dist && t_cls == worst_cls) {
        worst_dist = dist;
      }
    } else if (t_cls == max_terrain_) {
      worst_cls = max_terrain_;
      worst_dist = 0;
      // Can't get any worse than this
      break;
    }
    // Ignore unknwon
  } 

  return {worst_cls, 1/(worst_dist/params_.map_res + 0.01)};
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

  if (t_cls >= 0) {
    for (const auto& [overlap_n_id, overlap_loc] : overlapping_nodes) {
      auto overlap_n = &graph_.getNode(overlap_n_id);
      auto [worst_cls, worst_cost] = traceEdge(n->pos, overlap_n->pos);
      if (worst_cls <= t_cls) {
        graph_.addEdge({n, overlap_n, worst_cost, worst_cls});
      } else if (map_.at<uint8_t>(cv::Point(overlap_loc[0], overlap_loc[1])) <= t_cls) {
        // Add new intermediate node
        auto intermed_n = graph_.addNode({img2world(overlap_loc)});
        addNodeToVisibility(*intermed_n);
        auto edge_info = traceEdge(n->pos, intermed_n->pos);
        graph_.addEdge({n, intermed_n, edge_info.second, edge_info.first});
        edge_info = traceEdge(overlap_n->pos, intermed_n->pos);
        graph_.addEdge({overlap_n, intermed_n, edge_info.second, edge_info.first});
      }
    }
  }

  return n;
}

std::map<int, Eigen::Vector2f> TravMap::addNodeToVisibility(const TravGraph::Node& n) {
  // Get class of node location
  auto img_loc = world2img(n.pos);
  int node_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
  //std::cout << "+++++++++++" << std::endl;
  //std::cout << img_loc.transpose() << std::endl;
  //std::cout << node_cls << std::endl;

  std::map<int, Eigen::Vector2f> overlapping_nodes;

  // Choose delta such that ends of rays are within 0.5 cells
  float delta_t = 0.5 / (params_.vis_dist_m * params_.map_res);
  for (float theta=0; theta<2*pi; theta+=delta_t) {
    Eigen::Vector2f dir = {cos(theta), sin(theta)};

    for (float r=0; r<params_.map_res*params_.vis_dist_m; r+=0.5) {
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

      if (cell_cls != node_cls && cell_cls != 255) {
        break;
      }
    }
  }

  return overlapping_nodes;
}

cv::Mat TravMap::viz() const {
  cv::Mat scaled_map = map_.clone();
  // Rescale for increasing order of trav difficulty 0->255
  scaled_map *= 255. / max_terrain_;
  cv::Mat viz, cmapped_map;
  cv::applyColorMap(scaled_map, cmapped_map, cv::COLORMAP_PARULA);
  // Mask unknown regions
  cmapped_map.copyTo(viz, map_ < 255);
  // Draw origin
  auto origin_img = world2img({0, 0});
  cv::circle(viz, cv::Point(origin_img[0], origin_img[1]), 3, cv::Scalar(0, 0, 255), 2);

  // Draw nodes
  for (const auto& [node_id, node] : graph_.getNodes()) {
    auto node_img_pos = world2img(node.pos);
    cv::circle(viz, cv::Point(node_img_pos[0], node_img_pos[1]), 1, cv::Scalar(0, 255, 0), 2);
  }

  // Draw edges
  for (const auto& edge : graph_.getEdges()) {
    if (edge.cls >= max_terrain_) continue;
    auto node1_img_pos = world2img(edge.node1->pos);
    auto node2_img_pos = world2img(edge.node2->pos);
    cv::Scalar color;
    if (edge.cls == 0) {
      color = {0, 255, 0};
    } else if (edge.cls == 1 || edge.cls == 2) {
      color = {255, 0, 0};
    } else {
      color = {0, 0, 255};
    }
    color /= edge.cost;
    cv::line(viz, cv::Point(node1_img_pos[0], node1_img_pos[1]),
             cv::Point(node2_img_pos[0], node2_img_pos[1]), color);
  }
  
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
