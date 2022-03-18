#!/usr/bin/env python3

import rosbag
import argparse
import numpy as np
import cv2
from sensor_msgs.msg import Image

import matplotlib.pyplot as plt

class PanoPlan:
    def __init__(self, bag):
        self.bag_ = rosbag.Bag(bag)
        self.vfov_ = np.deg2rad(90.)

    def parse(self, num_panos = 1):
        count = 0
        for topic, msg, t in self.bag_.read_messages():
            if topic == '/os_node/llol_odom/pano':
                data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
                self.parse_pano(data)
                count += 1
                if count >= num_panos:
                    break
    
    def parse_pano(self, pano):
        dists = pano[:, :, 0].astype(np.float) / 512.
        elev_delta = self.vfov_ / (pano.shape[0] - 1)
        azi_delta = np.pi * 2 / pano.shape[1]
        elevs = np.arange(self.vfov_/2, -self.vfov_/2-0.0001, -elev_delta)
        alts = np.sin(elevs)[:, None] * dists

        #lr_delta = np.abs(dists - np.roll(dists, 1, axis=1)) / (azi_delta * dists)
        smoothed = cv2.filter2D(dists, cv2.CV_64F, np.ones((1, 7), dtype=np.float)/7)
        delta = np.abs(smoothed - dists) / (azi_delta * dists)

        self.viz_img(alts, 'alts')
        self.viz_img(smoothed, 'smoothed')
        self.viz_img(np.minimum(delta, 10), 'delta')
        plt.show()

    def viz_img(self, img, name=''):
        plt.figure()
        plt.imshow(img)
        plt.title(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    args = parser.parse_args()

    pp = PanoPlan(args.bag)
    pp.parse(1)
