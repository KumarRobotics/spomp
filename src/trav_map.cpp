#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include "spomp/trav_map.h"
#include "spomp/utils.h"

namespace spomp {

/**
 * @class TravMap
 * @brief Represents the map of a robot's traversal area
 */
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

/**
 * @brief Loads the classes from the given class_config and updates the terrain_lut and MAX_TERRAIN.
 *
 * @param class_config The ClassConfig object containing the class configurations.
 */
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

/**
 * @brief Load a static map based on the given map_config and class_config.
 *        If the map is dynamic, set the dynamic flag and return.
 *        Otherwise, load the static map and update the map center and class sem.
 *
 * @param map_config The map configuration.
 * @param class_config The class configuration.
 */
    void TravMap::loadStaticMap(const semantics_manager::MapConfig& map_config,
                                const semantics_manager::ClassConfig& class_config)
{
  map_ref_frame_.res = map_config.resolution;

  // Initialize dist_maps_
  int num_dist_maps = TravGraph::Edge::MAX_TERRAIN;
  if (params_.no_max_terrain_in_graph) {
    ++num_dist_maps;
  }

  dist_maps_.reserve(num_dist_maps);
  for (int terrain_ind=0; terrain_ind<num_dist_maps; ++terrain_ind) {
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

/**
 * @brief Updates the map with the given parameters and performs various operations on the map data.
 *
 * @param map The input map.
 * @param center The center position of the map.
 * @param other_maps Other maps for reference.
 */
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

  auto old_map_ref_frame = map_ref_frame_;
  map_ref_frame_.center = center;
  map_ref_frame_.setMapSizeFrom(map_);

  computeDistMaps();
  aerial_map_->updateMap(map_, dist_maps_, map_ref_frame_, other_maps);

  reweightGraph();
  rebuildVisibility(old_map_ref_frame);
  buildGraph();

  // Look through the reachability buffer and if we are now in the map,
  // take it off accordingly
  reach_buf_.remove_if([&](const auto& reach) -> bool
    {
      if (map_ref_frame_.imgPointInMap(map_ref_frame_.world2img(
              reach.getPose().translation())))
      {
        aerial_map_->updateLocalReachability(reach);
        graph_.updateLocalReachability(reach);
        return true;
      }
      return false;
    });
}

/**
 * @brief Get the path between two points in the traversal map.
 *
 * This function calculates the path between two points in the traversal map
 * using the A* algorithm implemented in the TravGraph class. The start and end
 * points are given as Eigen::Vector2f objects. If either of the points is out of
 * bounds or not visible, an empty list is returned.
 *
 * @param start_p The starting point in world coordinates.
 * @param end_p The ending point in world coordinates.
 * @return std::list<TravGraph::Node*> The list of nodes representing the path.
 *         If no valid path is found, an empty list is returned.
 */
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
    auto new_n = addNode(start_p);
    addEdge({n1, new_n, worst_edge.cost, worst_edge.cls});
    n1 = new_n;
  }
  if ((end_p - n2->pos).norm() > 2) {
    auto worst_edge = aerial_map_->traceEdge(end_p, n2->pos);
    auto img_loc = map_ref_frame_.world2img(end_p);
    auto new_n = addNode(end_p);
    addEdge({n2, new_n, worst_edge.cost, worst_edge.cls});
    n2 = new_n;
  }

  return getPath(*n1, *n2);
}

/**
 * @brief Updates the local reachability information for a given robot.
 *
 * @param reachability The reachability object containing scan data.
 * @param robot_id The ID of the robot.
 * @return True if successful, false otherwise.
 */
    bool TravMap::updateLocalReachability(const Reachability& reachability, int robot_id) {
  // Return true if we are at least reach_dist_thresh_m away from all prior reach
  auto is_far_from_prev_reach = [&](const Reachability& reach) {
    for (const auto& past_reach : reach_hist_[0]) {
      float dist = (past_reach.second.getPose().translation().head<2>() -
                    reach.getPose().translation().head<2>()).norm();
      if (dist < params_.reach_dist_thresh_m) {
        return false;
      }
    }
    return true;
  };

  if (robot_id < reach_hist_.size()) {
    if (robot_id != 0 || is_far_from_prev_reach(reachability)) {
      reach_hist_[robot_id].insert({reachability.getStamp(), reachability});
    }
  }

  if (map_ref_frame_.imgPointInMap(map_ref_frame_.world2img(
          reachability.getPose().translation())))
  {
    aerial_map_->updateLocalReachability(reachability);
    if (aerial_map_->haveNewTrav()) {
      reweightGraph();
      aerial_map_->setTravRead();
    }
    return graph_.updateLocalReachability(reachability);
  } else {
    // Reachability is not yet in the known map, add to buffer to add at a later time,
    // namely when we get a new (hopefully larger) map
    reach_buf_.push_back(reachability);
    return false;
  }
}

/**
 * @brief Get the path from the start node to the end node in the graph.
 *
 * @param start_n The start node.
 * @param end_n The end node.
 * @return std::list<TravGraph::Node*> The path from the start node to the end node.
 */
    std::list<TravGraph::Node*> TravMap::getPath(TravGraph::Node& start_n,
                                                 TravGraph::Node& end_n)
{
  auto path = graph_.getPath(&start_n, &end_n);
  auto final_path = path;
  if (final_path.size() < 1) {
    return final_path;
  }

  float original_cost = path.back()->cost;
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

/**
 * Prunes the given path by removing nodes that can be reached through shorter or lower-cost edges.
 * It returns a pruned path as a list of nodes.
 *
 * @param path The path to be pruned.
 * @return The pruned path as a list of nodes.
 */
    std::list<TravGraph::Node*> TravMap::prunePath(
    const std::list<TravGraph::Node*>& path) 
{
  std::list<TravGraph::Node*> pruned_path;
  if (path.size() < 1) return pruned_path;

  TravGraph::Node* last_node = path.front();
  TravGraph::Edge last_edge;
  double summed_cost = 0;
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

/**
 * \brief Computes distance maps for the given map.
 *
 * This function computes distance maps for the given map using the following steps:
 * 1. Creates a morphological kernel based on the maximum hole fill size specified in the parameters.
 * 2. Creates an unknown mask where the map values are equal to 255.
 * 3. Performs morphology opening on the unknown mask using the morphological kernel.
 * 4. Initializes a filtered map with all values set to 255.
 * 5. For each distance map in reverse order:
 *    a. Creates a traversal map by bitwise ORing the map values less than or equal to the current class threshold and the unknown mask.
 *    b. Performs morphology closing on the traversal map using the morphological kernel.
 *    c. Computes the distance transform of the traversal map and stores it in the current distance map.
 *    d. Sets the unknown mask values to 0 in the current distance map.
 *    e. Sets the filtered map values to the current class threshold where the distance map values are greater than 0.
 *    f. Decrements the class threshold.
 * 6. Updates the map with the filtered map.
 * 7. Ends the timer for computing distance maps.
 */
    void TravMap::computeDistMaps() {
  compute_dist_maps_t_->start();

  // Add 1 so if zero we still have valid morph element
  auto morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(
        params_.max_hole_fill_size_m * map_ref_frame_.res + 1,
        params_.max_hole_fill_size_m * map_ref_frame_.res + 1));
  
  cv::Mat unknown_mask = (map_ == 255);
  cv::morphologyEx(unknown_mask, unknown_mask, cv::MORPH_OPEN, morph_kernel);

  int t_cls = dist_maps_.size() - 1;
  cv::Mat filtered_map(map_.size(), CV_8UC1, cv::Scalar(255));
  for (auto it=dist_maps_.rbegin(); it!=dist_maps_.rend(); ++it) {
    cv::Mat trav_map;
    // 1 everywhere where we want to compute distance
    cv::bitwise_or(map_ <= t_cls, unknown_mask, trav_map);
    cv::morphologyEx(trav_map, trav_map, cv::MORPH_CLOSE, morph_kernel);
    cv::distanceTransform(trav_map, *it, cv::DIST_L2, cv::DIST_MASK_5);
    // Clear out unknown here.  We don't want to treat unknown as obstacle,
    // but we don't want to treat is as traversable either
    it->setTo(0, unknown_mask);
    filtered_map.setTo(t_cls, *it > 0);
    --t_cls;
  }

  map_ = filtered_map;

  compute_dist_maps_t_->end();
}

/**
* @brief Rebuilds the visibility map based on a new map reference frame.
*
* This function rebuilds the visibility map by taking into account a new map reference frame. The old visibility map is copied, and then intersected with the new frame to preserve any
* overlapping regions. The map is then updated to mark obstacles as -1 and unknown regions as -1. Next, the function iterates through each node in the graph and adds it to the visibility
* map. It also checks for any overlapping nodes and adds edges to improve connectivity. Finally, the function records the time it takes to rebuild the visibility map.
*
* @param old_mrf The old map reference frame.
* @return void
*/
    void TravMap::rebuildVisibility(const MapReferenceFrame& old_mrf) {
  rebuild_visibility_t_->start();

  cv::Mat old_vis_map = visibility_map_.clone();
  visibility_map_ = -cv::Mat::ones(map_.rows, map_.cols, CV_32SC1);

  auto intersect = old_mrf.computeIntersect(map_ref_frame_);

  if (!intersect.new_frame.empty()) {
    old_vis_map(intersect.old_frame).copyTo(visibility_map_(intersect.new_frame));
  }
  visibility_map_.setTo(-1, map_ == 255);

  // Now go through all nodes and add to the connectivity as needed
  // This also allows regions that are filled in by the quad later to improve their
  // connectivity.
  for (auto& [node_id, node] : graph_.getNodes()) {
    auto overlapping_nodes = addNodeToVisibility(node);

    for (const auto& [overlap_n_id, overlap_loc] : overlapping_nodes) {
      auto overlap_n = &graph_.getNode(overlap_n_id);
      auto worst_edge = aerial_map_->traceEdge(node.pos, overlap_n->pos);
      if (worst_edge.cls < TravGraph::Edge::MAX_TERRAIN) {
        addEdge({&node, overlap_n, worst_edge.cost, worst_edge.cls});
      }
    }
  }

  rebuild_visibility_t_->end();
}

/**
 * @brief Reweights the graph by updating edge weights and node visibility.
 *
 * This function reweights the graph by updating the weights of each edge.
 * If an edge is not locked and has not been experienced yet, it retrieves
 * the class and cost information from the aerial map and updates the edge.
 * Additionally, it checks the visibility of each node and sets the visibility map
 * for non-traversable nodes to -1.
 */
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

/**
 * @brief This function builds the graph for the TravMap object.
 *
 * The function iterates over the terrain classes from `start_cls` to `TravGraph::Edge::MAX_TERRAIN`
 * and performs the following steps for each terrain class:
 *
 * - Calculates the number of pixels that need to be covered, based on the condition `map_ <= t_cls`, using `cv::countNonZero`
 * - Clones the distance map of the current terrain class and creates a masked version of it using `cv::Mat::setTo`
 *   to set the masked pixels to zero (masked pixels are determined by `visibility_map_ >= 0`)
 * - Performs morphological opening on the masked distance map using a rectangular structuring element `morph_kernel`
 *   with size `(params_.min_region_size_m * map_ref_frame_.res + 1) x (params_.min_region_size_m * map_ref_frame_.res + 1)`,
 *   using `cv::morphologyEx`
 * - Checks if the number of non-zero pixels in the masked distance map is greater than a threshold value
 *   (`params_.unvis_start_thresh * std::pow(map_ref_frame_.res, 2)`).
 *   If the number is less than the threshold, the current terrain class is skipped.
 *
 * For each iteration, the function performs the following steps as long as the number of non-zero pixels in the masked distance map
 * is greater than another threshold value (`params_.unvis_stop_thresh * std::pow(map_ref_frame_.res, 2)`):
 *
 * - Resets the masked distance map by setting the masked pixels to zero again (`dist_map_masked.setTo`)
 * - Finds the location of the maximum value and its corresponding coordinates in the masked distance map using `cv::minMaxLoc`
 * - If the maximum value is zero, indicating that there are no more points to consider, the iteration is stopped
 * - Adds a new node to the graph at the world coordinates corresponding to the maximum value's coordinates in the image
 * - Marks the visibility of the maximum value's coordinates in the visibility map by setting the corresponding element
 *   to the ID of the newly added node (`visibility_map_.at<int32_t>(max_l) = n_ptr->id`)
 *
 * The function uses the `build_graph_t_` timer to record the execution time.
 */
    void TravMap::buildGraph() {
  build_graph_t_->start();

  double min_v, max_v;
  cv::Point min_l, max_l;

  // Add 1 so if zero we still have valid morph element
  auto morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(
        params_.min_region_size_m * map_ref_frame_.res + 1,
        params_.min_region_size_m * map_ref_frame_.res + 1));

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

      auto n_ptr = addNode(map_ref_frame_.img2world({max_l.x, max_l.y}));
      // Make absolutely sure that we don't pick the same pt again
      visibility_map_.at<int32_t>(max_l) = n_ptr->id;
    }
  }

  build_graph_t_->end();
}

/**
 * @brief Adds a new node to the TravMap.
 *
 * This function adds a new node representing a position on the map to the TravMap.
 *
 * @param pos The position of the node to be added, specified as a 2D vector of type Eigen::Vector2f.
 *
 * @note The added nodes can be used for various operations like path finding or traversal analysis.
 *
 * @see TravMap
 * @see TravMap::getNodeCount
 * @see TravMap::getNodePosition
 * @see TravMap::removeNode
 */
    TravGraph::Node* TravMap::addNode(const Eigen::Vector2f& pos) {
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
    if (worst_edge.cls < TravGraph::Edge::MAX_TERRAIN) {
      addEdge({n, overlap_n, worst_edge.cost, worst_edge.cls});
    }
  }

  return n;
}

/**
 * @class TravMap
 * @brief Represents a map of traveling routes.
 *
 * This class provides functionality to add edges to a map, which represents connections between locations.
 * Each edge is associated with a weight, representing the cost or distance between the connected locations.
 */
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

/**
 * @brief Adds a node to the visibility map and returns a map of overlapping nodes.
 *
 * This function adds a node to the visibility map and returns a map of overlapping nodes.
 * The visibility map is updated by casting rays from the given node's position in all directions.
 * The rays are terminated when they reach the specified visibility distance.
 * The overlapping nodes are stored in a map where the keys are the IDs of the overlapping nodes
 * and the values are the positions of the overlapping nodes.
 *
 * @param n The node to add to the visibility map.
 * @return std::map<int, Eigen::Vector2f> A map of overlapping nodes.
 */
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

/**
*
*/
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

/**
 * @brief Reset the locked status of the edges in the graph.
 *
 * This function iterates over each edge in the graph and sets the `is_locked` flag to `false`
 * for all edges that have a positive `cls` value.
 *
 * @note This function modifies the state of the graph.
 */
    void TravMap::resetGraphLocked() {
  for (auto& edge : graph_.getEdges()) {
    if (edge.cls > 0) {
      edge.is_locked = false;
    }
  }
}

/**
 * Generates a visualization of the TravMap.
 *
 * @return A cv::Mat object representing the visualization of the map.
 *
 * The visualization is generated by performing the following steps:
 * 1. If the map is empty, an empty cv::Mat is returned.
 * 2. A copy of the map is created and scaled to a range of 0-255, representing increasing difficulty.
 * 3. The scaled map is then color-mapped using the Parula colormap.
 * 4. Unknown regions in the map (values less than 255) are masked out.
 * 5. Nodes in the graph are drawn as green circles on the visualization.
 * 6. Edges in the graph are drawn as colored lines on the visualization.
 *    - If an edge is locked, it is drawn as a green line if cls is 0, and as a blue line if cls is non-zero.
 *    - If an edge is unlocked, a color is determined based on its cost and length, and drawn as a line on the visualization.
 * 7. The origin is drawn as a red circle on the visualization.
 */
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

/**
 * @brief Generate a visualization of the visibility map.
 *
 * This function generates a visualization of the visibility map. The visibility map is a matrix
 * that represents the visibility of each element in the graph. The generated visualization is a color
 * map that indicates the level of visibility for each element.
 *
 * @return A cv::Mat object representing the generated visualization.
 */
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
