#!/usr/bin/env python3

import numpy as np
import cv2

import rosbag
from sensor_msgs.msg import Image

class AerialMap:
    def __init__(self, path):
        self.color_ = None
        self.sem_ = None
        self.sem_viz_ = None
        self.intermed_ = None
        self.center_ = np.array([0., 0])

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
                elif topic == '/asoom/map_intermed_viz':
                    self.intermed_ = img
                    print("Loaded intermed")
            elif topic == '/asoom/map_sem_img_center':
                self.center_[0] = msg.point.x
                self.center_[1] = msg.point.x
                print("Loaded center")

    def show_color(self):
        cv2.imshow('color', self.color_)
        cv2.imshow('sem_viz', self.sem_viz_)
        cv2.waitKey(0)

if __name__ == '__main__':
    am = AerialMap('/media/ian/ResearchSSD/xview_collab/iapetus/twojackalquad4_asoomoutput.bag')
    am.show_color()
