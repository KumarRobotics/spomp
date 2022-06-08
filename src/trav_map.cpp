#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "spomp/trav_map.h"

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
  dist_maps_.reserve(max_terrain_+1);
  for (int terrain_ind=0; terrain_ind<=max_terrain_; ++terrain_ind) {
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
  int terrain_ind = 0;
  auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(
        params_.max_hole_fill_size_m * params_.map_res,
        params_.max_hole_fill_size_m * params_.map_res));

  for (auto& dist_map : dist_maps_) {
    cv::Mat trav_map = map_ <= terrain_ind;
    cv::morphologyEx(trav_map, trav_map, cv::MORPH_CLOSE, kernel);
    cv::distanceTransform(trav_map, dist_map, cv::DIST_L2, cv::DIST_MASK_5);
    ++terrain_ind;
  }
}

void TravMap::moveVisibilityGraph(const Eigen::Vector2f& old_center) {
  if (visibility_map_.empty()) {
    visibility_map_ = cv::Mat::zeros(map_.rows, map_.cols, CV_16UC1);
    return;
  }

  cv::Mat old_viz_map = visibility_map_;
  auto old_center_in_new = world2img(old_center);

  visibility_map_ = cv::Mat::zeros(map_.rows, map_.cols, CV_16UC1);
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
}

void TravMap::buildGraph() {
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
    auto cls = map_.at<uint8_t>(sample_pt[0], sample_pt[1]);
    float dist = 0;
    if (cls < 255) {
      dist = dist_maps_[cls].at<uint16_t>(sample_pt[0], sample_pt[1]);
    
      if (cls > worst_cls) {
        worst_cls = cls;
        worst_dist = dist;
      } else if (dist < worst_dist && cls == worst_cls) {
        worst_dist = dist;
      }
    }
  } 

  return {worst_cls, worst_dist/params_.map_res};
}

std::set<int> TravMap::addNode(const TravGraph::Node& n) {
  return {};
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

} // namespace spomp
