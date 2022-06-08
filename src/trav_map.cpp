#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "spomp/trav_map.h"
#include "spomp/utils.h"

namespace spomp {

TravMap::TravMap(const Params& p) : params_(p) {
  loadTerrainLUT();
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
    terrain_lut_.at<uint8_t>(0) = 1;
    terrain_lut_.at<uint8_t>(1) = 0;
  }

  // Initialize dist_maps_
  dist_maps_.reserve(max_terrain_);
  for (int terrain_ind=0; terrain_ind<max_terrain_; ++terrain_ind) {
    dist_maps_.emplace_back();
  }
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
  auto old_center = map_center_;
  map_center_ = center;

  computeDistMaps();
  moveVisibilityGraph(old_center);
  reweightGraph();
  buildGraph();
}

void TravMap::computeDistMaps() {
  int t_cls = 0;
  auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(
        params_.max_hole_fill_size_m * params_.map_res,
        params_.max_hole_fill_size_m * params_.map_res));

  for (auto& dist_map : dist_maps_) {
    cv::Mat trav_map = map_ <= t_cls;
    //cv::morphologyEx(trav_map, trav_map, cv::MORPH_CLOSE, kernel);
    cv::distanceTransform(trav_map, dist_map, cv::DIST_L2, cv::DIST_MASK_5);
    ++t_cls;
  }
}

void TravMap::moveVisibilityGraph(const Eigen::Vector2f& old_center) {
  if (visibility_map_.empty()) {
    visibility_map_ = -cv::Mat::ones(map_.rows, map_.cols, CV_32SC1);
    return;
  }

  cv::Mat old_viz_map = visibility_map_;
  auto old_center_in_new = world2img(old_center);

  visibility_map_ = -cv::Mat::ones(map_.rows, map_.cols, CV_32SC1);
  old_viz_map.copyTo(visibility_map_(cv::Rect(
          old_center_in_new[0] - old_viz_map.rows/2,
          old_center_in_new[1] - old_viz_map.cols/2,
          old_viz_map.rows,
          old_viz_map.cols
          )));
}

void TravMap::reweightGraph() {
  for (auto& edge : graph_.getEdges()) {
    auto edge_info = traceEdge(edge.node1->pos, edge.node2->pos);
    edge.cls = edge_info.first;
    edge.cost = 1./(edge_info.second + 0.01);
  }

  for (auto& [node_id, node] : graph_.getNodes()) {
    auto img_loc = world2img(node.pos);
    int node_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
    if (node_cls == max_terrain_) {
      // Node is no longer traversable, sad
      visibility_map_.setTo(-1, visibility_map_ == node_id);
    }
  }
}

void TravMap::buildGraph() {
  double min_v, max_v;
  cv::Point min_l, max_l;

  for (int t_cls=0; t_cls<max_terrain_; ++t_cls) {
    int num_to_cover = cv::countNonZero(map_ <= t_cls);
    cv::Mat dist_map_masked = dist_maps_[t_cls];
    do {
      // Ignore points already visible
      dist_map_masked.setTo(0, visibility_map_ >= 0);
      // Select next best node location
      cv::minMaxLoc(dist_maps_[t_cls], &min_v, &max_v, &min_l, &max_l);
      TravGraph::Node* n = graph_.addNode({img2world({max_l.x, max_l.y})});
      auto overlapping_nodes = addNode(*n);
      std::cout << "=============" << std::endl;
      std::cout << "target cls: " << t_cls << std::endl;
      std::cout << n->pos.transpose() << std::endl;
      std::cout << max_v << std::endl;
      std::cout << max_l << std::endl;
      std::cout << cv::countNonZero(visibility_map_ >= 0) << std::endl;
      std::cout << num_to_cover << std::endl;
    } while (cv::countNonZero(visibility_map_ >= 0) < 0.99 * num_to_cover);
  }
}

std::pair<int, float> TravMap::traceEdge(const Eigen::Vector2f& n1, 
    const Eigen::Vector2f& n2)
{
  auto img_pt1 = world2img(n1);
  auto img_pt2 = world2img(n2);
  float dist = (img_pt1 - img_pt2).norm();
  Eigen::Vector2f dir = (img_pt2 - img_pt1).normalized();

  int worst_cls = max_terrain_;
  float worst_dist = 0;

  for (float cur_dist=0; cur_dist<dist; cur_dist+=0.5) {
    Eigen::Vector2f sample_pt = img_pt1 + dir*cur_dist;
    auto t_cls = map_.at<uint8_t>(cv::Point(sample_pt[0], sample_pt[1]));
    float dist = 0;
    if (t_cls < max_terrain_) {
      dist = dist_maps_[t_cls].at<uint16_t>(cv::Point(sample_pt[0], sample_pt[1]));
    
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
  } 

  return {worst_cls, worst_dist/params_.map_res};
}

std::map<int, Eigen::Vector2f> TravMap::addNode(const TravGraph::Node& n) {
  // Get class of node location
  std::cout << "+++++++++++" << std::endl;
  auto img_loc = world2img(n.pos);
  std::cout << img_loc.transpose() << std::endl;
  int node_cls = map_.at<uint8_t>(cv::Point(img_loc[0], img_loc[1]));
  std::cout << node_cls << std::endl;

  std::map<int, Eigen::Vector2f> overlapping_nodes;

  // Choose delta such that ends of rays are within 0.5 cells
  float delta_t = 0.5 / (params_.vis_dist_m * params_.map_res);
  for (float theta=0; theta<2*pi; theta+=delta_t) {
    for (float r=0; r<params_.map_res*params_.vis_dist_m; r+=0.5) {
      Eigen::Vector2f img_cell = img_loc + Eigen::Vector2f(cos(theta), sin(theta))*r;
      if ((img_cell.array() < 0).any() || 
          (img_cell.array() >= Eigen::Vector2f(map_.rows, map_.cols).array()).any()) {
        // We have left the image
        continue;
      }

      int cell_cls = map_.at<uint8_t>(cv::Point(img_cell[0], img_cell[1]));
      auto& vis_cell = visibility_map_.at<int32_t>(cv::Point(img_cell[0], img_cell[1]));

      // We want cells to be visible from more difficult regions
      if (cell_cls < max_terrain_ && (vis_cell < 0 || cell_cls >= node_cls)) {
        // Either this cell is not yet visbile, or is less traversable than target node
        if (vis_cell >= 0) {
          overlapping_nodes.emplace(vis_cell, img_cell);
        }
        vis_cell = n.id;
      } else if (cell_cls != 255) {
        // Include ray endpoint
        // This helps prevent infinite loops, so we don't select same point again
        vis_cell = n.id;
        // End ray
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
