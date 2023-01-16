#!/usr/bin/env python3

import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import rospy
import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from spomp.msg import LocalReachabilityArray

class AerialMap:
    def __init__(self, path):
        self.color_ = None
        self.sem_ = None
        self.sem_viz_ = None
        self.intermed_ = None
        self.origin_ = np.array([0., 0])
        self.map_size_ = np.array([0., 0])
        self.scale_ = 2
        self.reach_proj_ = None
        self.no_reach_proj_ = None

        self.model_ = None

        for topic, msg, t in rosbag.Bag(path, 'r').read_messages():
            if 'Image' in str(msg.__class__):
                dtype = np.uint8
                if msg.encoding.startswith('32F'):
                    dtype = np.float32
                img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
                if topic == '/asoom/map_color_img':
                    self.color_ = img
                    print("Loaded color")
                elif topic == '/asoom/map_sem_img':
                    self.sem_ = img
                    self.map_size_ = np.array(img.shape, dtype=np.float32)[:2]
                    print("Loaded sem")
                elif topic == '/asoom/map_sem_img_viz':
                    self.sem_viz_ = img
                    print("Loaded sem viz")
                elif topic == '/asoom/map_intermed_img':
                    self.intermed_ = img
                    print("Loaded intermed")
            elif topic == '/asoom/map_sem_img_center':
                self.origin_[0] = msg.point.x
                self.origin_[1] = msg.point.y
                print("Loaded center")

        self.new_pts_trav_ = True
        self.trav_img_ = np.zeros(self.sem_.shape, dtype=np.uint8)
        self.no_trav_img_ = np.zeros(self.sem_.shape, dtype=np.uint8)

        self.map_pub_ = rospy.Publisher('~map', Image, queue_size=10)
        self.map_click_sub_ = rospy.Subscriber('~map_mouse_left', Point, self.map_click_cb)

        self.est_trav_pub_ = rospy.Publisher('~est_trav', Image, queue_size=10)
        self.est_trav_click_sub_ = rospy.Subscriber('~est_trav_mouse_left', Point, 
                self.est_trav_click_cb)

        self.pub_timer_ = rospy.Timer(rospy.Duration(1), self.publish_map)

    def parse_reachability(self, reachability):
        if self.reach_proj_ is None: 
            # assume projection intrinsics are always the same, so just have to 
            # compute once
            thetas = np.arange(reachability.reachability.angle_min, 
                               reachability.reachability.angle_max, 
                               reachability.reachability.angle_increment)[:, None]
            pts = np.hstack((np.cos(thetas), np.sin(thetas)))
            ranges = np.arange(0, reachability.reachability.range_max-1, 1/self.scale_)
            self.reach_proj_ = pts[None,:,:] * ranges[:,None,None]/np.max(ranges)
            self.no_reach_proj_ = pts

        pos = np.array([reachability.pose.x, reachability.pose.y])
        theta = reachability.pose.theta
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        trav_pts = self.reach_proj_ * np.array(reachability.reachability.ranges)[None,:,None]

        # select only rays terminating with obstacle
        obs_rays = np.array(reachability.reachability.intensities, dtype=bool)
        no_trav_ranges = np.array(reachability.reachability.ranges)[obs_rays]

        # select the next chunk of meters after the obstacle
        no_trav_ranges = no_trav_ranges[None,:] + np.arange(0, 3, 1)[:,None]
        no_trav_pts = self.no_reach_proj_[None,obs_rays] * no_trav_ranges[:,:,None]

        return (pos + (rot @ trav_pts.reshape(-1, 2).transpose()).transpose(), 
                pos + (rot @ no_trav_pts.reshape(-1, 2).transpose()).transpose())

    def load_trav(self, path):
        comb_trav_img = np.zeros(self.sem_.shape, dtype=np.int)
        for topic, msg, t in rosbag.Bag(path, 'r').read_messages():
            for reachability in msg.reachabilities:
                reach_pts, no_reach_pts = self.parse_reachability(reachability)
                reach_pts_px = self.world2img(reach_pts)
                comb_trav_img[reach_pts_px[:, 0], reach_pts_px[:, 1]] += 1
                no_reach_pts_px = self.world2img(no_reach_pts)
                comb_trav_img[no_reach_pts_px[:, 0], no_reach_pts_px[:, 1]] -= 1
            break

        self.trav_img_ = (comb_trav_img>3).astype(np.uint8)
        self.no_trav_img_ = (comb_trav_img<0).astype(np.uint8)

    def world2img(self, world_pos):
        return ((-world_pos + self.origin_) * self.scale_ +
                self.map_size_/2).astype(np.int)

    def img2world(self, img_pos):
        return -(((img_pos -
                self.map_size_/2)/self.scale_) - self.origin_)

    def get_trav_color(self, is_trav):
        if is_trav:
            return (0, 255, 0)
        else:
            return (0, 0, 255)

    def get_sample_pts(self):
        X = np.empty((0, 16))
        y = np.empty(0, dtype=np.uint8)

        X = np.vstack((X, self.intermed_[self.trav_img_[:,:,0]>0, :]))
        y = np.tile(1, int(np.sum(self.trav_img_)))
        X = np.vstack((X, self.intermed_[self.no_trav_img_[:,:,0]>0, :]))
        y = np.concatenate((y, np.tile(0, int(np.sum(self.no_trav_img_)))))

        return X, y
    
    def fit_model(self):
        X, y = self.get_sample_pts()

        print("Fitting model")
        self.model_ = MLPClassifier(random_state=1, max_iter=1000, alpha=5).fit(X, y)
        print("Model fit")

    def pub_rgb(self, image, publisher):
        msg = Image()
        msg.encoding = "bgr8"
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.step = msg.width * 3
        msg.data = image.tobytes()
        publisher.publish(msg)

    def publish_map(self, timer=None):
        annotated_map = self.color_.copy()

        cv2.rectangle(annotated_map, (0, 0), (20, 20), self.get_trav_color(self.new_pts_trav_), -1)
        annotated_map[self.trav_img_[:,:,0]>0, :] = np.array([0,255,0])
        annotated_map[self.no_trav_img_[:,:,0]>0, :] = np.array([0,0,255])

        self.pub_rgb(annotated_map, self.map_pub_)
        if self.model_ is None:
            self.pub_rgb(annotated_map, self.est_trav_pub_)

    def map_click_cb(self, click_pt_msg):
        if click_pt_msg.x < 20 and click_pt_msg.y < 20:
            self.new_pts_trav_ = not self.new_pts_trav_
            self.publish_map()
            return

        pt = np.array([click_pt_msg.x, click_pt_msg.y], dtype=np.int16)
        if self.new_pts_trav_:
            cv2.circle(self.trav_img_, pt, 3, 1, thickness=-1)
            cv2.circle(self.no_trav_img_, pt, 3, 0, thickness=-1)
        else:
            cv2.circle(self.trav_img_, pt, 3, 0, thickness=-1)
            cv2.circle(self.no_trav_img_, pt, 3, 1, thickness=-1)
        self.publish_map()

    def est_trav_click_cb(self, click_pt_msg):
        self.fit_model()
        if self.model_ is None:
            print("No model yet")
            return

        print("Inferring trav")
        nonzero_pts = (self.sem_[:,:,0] != 255)
        X = self.intermed_[nonzero_pts, :]

        probs = self.model_.predict_proba(X)
        print("Trav inferred")

        annotated_map = self.color_.copy()
        probs_img = np.zeros(self.sem_.shape)
        probs_img[nonzero_pts] = probs[:, 0, None]

        # colormap
        probs_img = (probs_img*255).astype(np.uint8)
        probs_color = cv2.applyColorMap(probs_img, cv2.COLORMAP_JET)
        annotated_map = cv2.addWeighted(annotated_map, 0.5, probs_color, 0.5, 0.0)

        self.pub_rgb(annotated_map, self.est_trav_pub_)

if __name__ == '__main__':
    rospy.init_node("aerial_context_test")
    am = AerialMap('/media/ian/ResearchSSD/xview_collab/iapetus/twojackalquad4_asoomoutput_cityscapes.bag')
    am.load_trav('/media/ian/ResearchSSD/xview_collab/iapetus/twojackalquad4_spompreachability.bag')
    rospy.spin()
