#!/usr/bin/env python3

import rosbag
import argparse
import numpy as np
from vedo import Points, show
from scipy import ndimage
import cv2
from skimage import morphology
from sensor_msgs.msg import Image

import matplotlib.pyplot as plt

class PanoPlan:
    def __init__(self, bag):
        self.bag_ = rosbag.Bag(bag)
        self.vfov_ = np.deg2rad(90.)

    def parse(self, num_panos = np.inf, start_ind = 0):
        count = 0
        for topic, msg, t in self.bag_.read_messages():
            if topic == '/os_node/llol_odom/pano':
                data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
                if count >= start_ind:
                    self.parse_pano(data)
                count += 1
                if count >= num_panos + start_ind:
                    break
    
    def parse_pano(self, pano):
        dists = pano[:, :, 0].astype(np.float32) / 512.

        #fill holes

        elev_delta = self.vfov_ / (pano.shape[0] - 1)
        azi_delta = np.pi * 2 / pano.shape[1]
        elevs = np.arange(self.vfov_/2, -self.vfov_/2-0.0001, -elev_delta)
        azis = np.arange(0, np.pi*2, azi_delta)
        zs = np.sin(elevs)[:, None] * dists
        ranges = np.cos(elevs)[:, None] * dists
        xs = np.cos(azis) * ranges
        ys = np.sin(azis) * ranges

        # using nan instead of zero makes median work better
        dists[dists == 0] = np.nan
        ranges[ranges == 0] = np.nan
        alts = zs.copy()
        alts[alts == 0] = np.nan

        smoothed_dist = np.zeros(alts.shape, dtype=np.float32)
        for ind, row in enumerate(dists):
            avg_dist = np.nanmedian(ranges[ind, :])
            if np.isnan(avg_dist):
                continue
            window_size = 0.5 / (avg_dist * azi_delta)
            smoothed_dist[ind, :] = ndimage.median_filter(row, np.maximum(1, int(window_size)))

        alt_delta = np.maximum(np.abs(np.roll(smoothed_dist, -2, axis=1) - np.roll(smoothed_dist, 2, axis=1)) - 0.03, 0) / (azi_delta * ranges**2) * np.abs(np.sin(elevs[:, None]))

        smoothed_vert_range = np.cos(elevs)[:, None] * smoothed_dist
        smoothed_vert_alt = np.sin(elevs)[:, None] * smoothed_dist
        #smoothed_vert_alt = ndimage.median_filter(alts, (1, 1))
        #smoothed_vert_range = ndimage.median_filter(ranges, (1, 1))
        #find short vertical obstacles
        alt_delta_vert = np.maximum(np.abs((smoothed_vert_alt - np.roll(smoothed_vert_alt, 1, axis=0))) - 0.03 * (1 - np.abs(np.sin(elevs[:, None]))), 0) / \
                np.abs(smoothed_vert_range - np.roll(smoothed_vert_range, 1, axis=0))
        #find shallower obstacles
        alt_delta_vert += np.maximum(np.abs((smoothed_vert_alt - np.roll(smoothed_vert_alt, 3, axis=0))) - 0.03 * (1 - np.abs(np.sin(elevs[:, None]))), 0) / \
                np.abs(smoothed_vert_range - np.roll(smoothed_vert_range, 3, axis=0))

        obs = np.maximum(alt_delta, alt_delta_vert) > 0.3
        obs = np.nan_to_num(obs)
        obs = morphology.remove_small_objects(obs, 10)

        #inflate along azimuth
        obs_inf = np.zeros(obs.shape, dtype=np.float32)
        for ind, row in enumerate(obs):
            avg_dist = np.nanmedian(ranges[ind, :])
            if np.isnan(avg_dist):
                continue
            window_size = 0.5 / (avg_dist * azi_delta)
            obs_inf[ind, :] = cv2.dilate(row[None, :].astype(np.uint8), np.ones([1, int(window_size)]))
        #inflate along altitude
        for row_ind, row in enumerate(obs.T):
            initial_range = 0
            for col_ind, pt in enumerate(row):
                if pt > 0:
                    initial_range = ranges[col_ind, row_ind]
                if initial_range != 0 and np.abs(initial_range - ranges[col_ind, row_ind]) < 0.5:
                    obs_inf[col_ind, row_ind] = 1
        obs_inf = np.maximum(obs.astype(np.float32)*2, obs_inf)

        pc_points = Points([xs.flatten(), ys.flatten(), zs.flatten()]).cmap("rainbow", obs_inf.flatten())
        show(pc_points, resetcam=False, camera={'pos':(0,0,30)}).close()

        #self.viz_img(alts, 'alts')
        #self.viz_img(smoothed_vert_alt, 'smoothed')
        #self.viz_img(np.minimum(alt_delta, 1), 'alt_delta')
        #self.viz_img(np.minimum(alt_delta_vert, 1), 'alt_delta_vert')
        #self.viz_img(obs_inf, 'obs')
        #plt.show()

    def viz_img(self, img, name=''):
        plt.figure()
        plt.imshow(img)
        plt.title(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    args = parser.parse_args()

    pp = PanoPlan(args.bag)
    pp.parse(start_ind = 1)
