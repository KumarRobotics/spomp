#!/usr/bin/env python3

import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import rospy
import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

class AerialMap:
    def __init__(self, path):
        self.color_ = None
        self.sem_ = None
        self.sem_viz_ = None
        self.intermed_ = None
        self.center_ = np.array([0., 0])

        self.trav_edges_ = []
        self.new_edges_trav_ = True
        self.last_pt_ = None

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
                    print("Loaded sem")
                elif topic == '/asoom/map_sem_img_viz':
                    self.sem_viz_ = img
                    print("Loaded sem viz")
                elif topic == '/asoom/map_intermed_img':
                    self.intermed_ = img
                    print("Loaded intermed")
            elif topic == '/asoom/map_sem_img_center':
                self.center_[0] = msg.point.x
                self.center_[1] = msg.point.x
                print("Loaded center")

        self.map_pub_ = rospy.Publisher('~map', Image, queue_size=10)
        self.map_click_sub_ = rospy.Subscriber('~map_mouse_left', Point, self.map_click_cb)

        self.est_trav_pub_ = rospy.Publisher('~est_trav', Image, queue_size=10)
        self.est_trav_click_sub_ = rospy.Subscriber('~est_trav_mouse_left', Point, 
                self.est_trav_click_cb)

        self.pub_timer_ = rospy.Timer(rospy.Duration(1), self.publish_map)

    def get_trav_color(self, is_trav):
        if is_trav:
            return (0, 255, 0)
        else:
            return (0, 0, 255)

    def get_sample_pts(self):
        X = np.empty((0, 16*2))
        y = np.empty(0, dtype=np.uint8)

        for edge in self.trav_edges_:
            descriptor0 = self.intermed_[edge[1][1], edge[1][0]]
            descriptor1 = self.intermed_[edge[2][1], edge[2][0]]
            X = np.vstack((X, np.concatenate((descriptor0, descriptor1))))
            X = np.vstack((X, np.concatenate((descriptor1, descriptor0))))

            y = np.append(y, (edge[0], edge[0]))

        return X, y
    
    def fit_model(self):
        X, y = self.get_sample_pts()

        print("Fitting model")
        self.model_ = MLPClassifier(random_state=1, max_iter=1000).fit(X, y)
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

        cv2.rectangle(annotated_map, (0, 0), (20, 20), self.get_trav_color(self.new_edges_trav_), -1)
        for edge in self.trav_edges_:
            cv2.line(annotated_map, tuple(edge[1]), tuple(edge[2]), 
                    self.get_trav_color(edge[0]), 1)

        self.pub_rgb(annotated_map, self.map_pub_)
        if self.model_ is None:
            self.pub_rgb(annotated_map, self.est_trav_pub_)

    def map_click_cb(self, click_pt_msg):
        if click_pt_msg.x < 20 and click_pt_msg.y < 20:
            self.new_edges_trav_ = not self.new_edges_trav_
            self.publish_map()
            return

        pt = np.array([click_pt_msg.x, click_pt_msg.y], dtype=np.int16)
        if self.last_pt_ is None:
            self.last_pt_ = pt
        else:
            self.trav_edges_.append((self.new_edges_trav_, pt, self.last_pt_))
            self.last_pt_ = None
            self.publish_map()
            self.fit_model()

    def est_trav_click_cb(self, click_pt_msg):
        if self.model_ is None:
            print("No model yet")
            return

        print("Inferring trav")
        start_desc = self.intermed_[int(click_pt_msg.y), int(click_pt_msg.x)]
        nonzero_pts = (self.sem_[:,:,0] != 255)
        end_desc = self.intermed_[nonzero_pts, :]
        X = np.hstack((np.tile(start_desc, (end_desc.shape[0], 1)), end_desc))

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
    rospy.spin()
