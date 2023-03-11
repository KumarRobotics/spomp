#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from collections import OrderedDict
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseArray, Pose

class FindObjectGoalGen:
    def __init__(self):
        self.last_map_stamp_ = 0
        self.map_size_ = np.array([0., 0])
        self.scale_ = 0
        self.class_list_ = [0, 6, 3]

        self.scale_ = rospy.get_param("~resolution", 2)
        self.merge_dist_ = rospy.get_param("~merge_dist", 10) * self.scale_
        self.min_size_ = (rospy.get_param("~min_size", 2) * self.scale_) ** 2
        self.region_size_ = rospy.get_param("~region_size", 7) * self.scale_

        self.map_sub_ = rospy.Subscriber("~map", GridMap, self.map_cb)
        self.goal_viz_pub_ = rospy.Publisher("~goal_viz", Image, queue_size=1)
        self.target_goals_pub_ = rospy.Publisher("~target_goals", PoseArray, queue_size=1)

    def world2img(self, world_pos):
        return ((-world_pos + self.origin_) * self.scale_ +
                self.map_size_/2).astype(np.int)

    def img2world(self, img_pos):
        return -(((img_pos -
                self.map_size_/2)/self.scale_) - self.origin_)

    def map_cb(self, map_msg):
        if map_msg.info.header.stamp.to_nsec() <= self.last_map_stamp_:
            return
        self.last_map_stamp_ = map_msg.info.header.stamp.to_nsec()

        if map_msg.info.length_x < 1 or map_msg.info.length_y < 1:
            return

        center = map_msg.info.pose.position

        layer_ind = map_msg.layers.index("semantics")
        sem_layer = map_msg.data[layer_ind]
        map_img = np.nan_to_num(np.array(sem_layer.data), nan=255).astype(np.uint8).reshape(
                sem_layer.layout.dim[0].size, sem_layer.layout.dim[1].size)
        map_img = map_img.transpose()

        self.scale_ = 1./map_msg.info.resolution
        self.origin_ = np.array([center.x, center.y])
        self.map_size_ = np.array(map_img.shape, dtype=np.float32)[:2]

        self.find_goals(map_img)

    def compute_dist_map(self, map_img, cls_list):
        road_map = np.isin(map_img, cls_list)
        # 0 everywhere we have obstacle
        obstacle_map_inv = np.logical_or(road_map, map_img == 255) 
        road_map = cv2.morphologyEx(road_map.astype(np.uint8), cv2.MORPH_CLOSE, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        obstacle_map_inv = cv2.morphologyEx(obstacle_map_inv.astype(np.uint8), cv2.MORPH_CLOSE, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        # make sure we don't fill in small things like cars which are legit
        road_map = np.logical_and(road_map, obstacle_map_inv).astype(np.uint8)

        return cv2.distanceTransform(road_map, cv2.DIST_L2, cv2.DIST_MASK_5)

    def detect_roi(self, map_img):
        car_locations = (map_img == 4).astype(np.uint8)*255
        # remove points with very small area
        car_locations_filtered = cv2.morphologyEx(car_locations, cv2.MORPH_OPEN, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        # merge blobs
        car_locations_filtered = cv2.morphologyEx(car_locations_filtered, cv2.MORPH_CLOSE, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                    (self.merge_dist_, self.merge_dist_)))

        # detect blobs
        if cv2.__version__.startswith("4."):
            blobs, _ = cv2.findContours(car_locations_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            _, blobs, _ = cv2.findContours(car_locations_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        blob_centers = []
        for blob in blobs:
            moments = cv2.moments(blob)
            area = moments['m00']
            if area >= self.min_size_:
                blob_centers.append(self.img2world(np.array([
                    moments['m01']/area, moments['m10']/area])))
        
        return blob_centers

    def choose_goals(self, blob_centers, map_img):
        goals = np.zeros((0, 2)) 
        for goal in blob_centers:
            # compute search square
            goal_region_min_corner = np.maximum(0, self.world2img(goal) - 
                    int(self.region_size_))
            goal_region_max_corner = np.minimum(np.array(map_img.shape[:2]), 
                    self.world2img(goal) + int(self.region_size_))

            for cls_num in range(1, len(self.class_list_)+1):
                distance_map = self.compute_dist_map(map_img, self.class_list_[:cls_num])
                # Ok, find safest location in goal region
                local_dist_map = distance_map[goal_region_min_corner[0]:goal_region_max_corner[0],
                                              goal_region_min_corner[1]:goal_region_max_corner[1]]
                best_pt = np.unravel_index(local_dist_map.argmax(), local_dist_map.shape)
                if local_dist_map[best_pt[0], best_pt[1]] > 0:
                    # the best point is in a valid region
                    best_pt += goal_region_min_corner
                    goals = np.vstack([goals, self.img2world(best_pt)[None,:]])
                    break

        return goals

    def find_goals(self, map_img):
        blob_centers = self.detect_roi(map_img) 
        goals = self.choose_goals(blob_centers, map_img)
        rospy.loginfo(f"Found {goals.shape[0]} goals")
        self.pub_viz(map_img, blob_centers, goals)
        self.pub_goals(goals)

    def pub_goals(self, goals):
        goals_msg = PoseArray()
        goals_msg.header.stamp = rospy.Time.now()
        goals_msg.header.frame_id = "map"
        for goal in goals:
            goal_msg = Pose()
            goal_msg.orientation.w = 1
            goal_msg.position.x = goal[0]
            goal_msg.position.y = goal[1]
            goals_msg.poses.append(goal_msg)
        self.target_goals_pub_.publish(goals_msg)

    def pub_viz(self, viz_img, blob_centers, goals):
        viz = cv2.cvtColor(viz_img.copy()*20, cv2.COLOR_GRAY2BGR)
        viz[viz[:,:,0] == 4*20, :] = np.array([255, 255, 0])
        for blob_w in blob_centers:
            blob = self.world2img(blob_w)
            min_b = np.flip((blob - self.region_size_).astype(np.int))
            max_b = np.flip((blob + self.region_size_).astype(np.int))
            viz = cv2.rectangle(viz, tuple(min_b), tuple(max_b), (0, 0, 255), 2)
        for target_w in goals:
            target = np.flip(self.world2img(target_w))
            viz = cv2.circle(viz, tuple(target), 4, (0, 255, 0), -1)

        viz_msg = Image()
        viz_msg.encoding = "bgr8"
        viz_msg.height = viz_img.shape[0]
        viz_msg.width = viz_img.shape[1]
        viz_msg.step = viz_msg.width * 3
        viz_msg.data = viz.tobytes()
        self.goal_viz_pub_.publish(viz_msg)

if __name__ == '__main__':
    rospy.init_node('find_object_goal_gen')
    fogg = FindObjectGoalGen()
    rospy.spin()
