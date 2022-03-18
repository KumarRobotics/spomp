#!/usr/bin/env python3

import rosbag
import argparse
import numpy as np
#from vedo import Points, show
from scipy import ndimage
import cv2
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from skimage import morphology
from sensor_msgs.msg import Image

import matplotlib.pyplot as plt

class PanoPlan:
    def __init__(self, bag):
        self.bag_ = rosbag.Bag(bag)
        self.vfov_ = np.deg2rad(90.)

        rospy.init_node('pano_plan')
        self.viz_pub_ = rospy.Publisher('~viz', PointCloud2, queue_size=1)

    def parse(self, num_panos = np.inf, start_ind = 0):
        count = 0
        rate = rospy.Rate(10)
        for topic, msg, t in self.bag_.read_messages():
            if rospy.is_shutdown():
                break
            if topic == '/os_node/llol_odom/pano':
                data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
                if count >= start_ind:
                    self.parse_pano(data)
                    print(count)
                    rate.sleep()
                count += 1
                if count >= num_panos + start_ind:
                    break
    
    def parse_pano(self, pano):
        dists = pano[:, :, 0].astype(np.float32) / 512.

        #helpful LUTs
        elev_delta = self.vfov_ / (pano.shape[0] - 1)
        azi_delta = np.pi * 2 / pano.shape[1]
        elevs = np.arange(self.vfov_/2, -self.vfov_/2-0.0001, -elev_delta)
        azis = np.arange(0, np.pi*2, azi_delta)
        
        #fill holes
        robot_elev = 0.4
        for row_ind, row in enumerate(dists):
            if elevs[row_ind] > np.deg2rad(-5):
                #ignore the top half
                continue
            nonzero_inds = np.nonzero(row)[0]
            if len(nonzero_inds) < len(row)/10:
                # not meaningful enough to fill in
                if elevs[row_ind] < np.deg2rad(-5):
                    dists[row_ind, :] = robot_elev / np.sin(-elevs[row_ind])
                else:
                    dists[row_ind, :] = 0 
                continue
            else:
                robot_elev = np.median(np.sin(-elevs[row_ind]) * row[nonzero_inds])

            for ind, dist in enumerate(row):
                if dist == 0:
                    before = nonzero_inds[nonzero_inds < ind]
                    after = nonzero_inds[nonzero_inds > ind]
                    if len(before) == 0:
                        if after[0] < 100:
                            dists[row_ind, ind] = row[after[0]]
                    elif len(after) == 0:
                        if len(row) - before[-1] < 100:
                            dists[row_ind, ind] = row[before[-1]]
                    else:
                        #linear interp
                        if after[0] - before[-1] < 100:
                            dists[row_ind, ind] = ((ind - before[-1]) * row[after[0]] + (after[0] - ind) * row[before[-1]]) / (after[0] - before[-1])

        zs = np.sin(elevs)[:, None] * dists
        ranges = np.cos(elevs)[:, None] * dists
        xs = np.cos(azis) * ranges
        ys = np.sin(azis) * ranges

        # using nan instead of zero makes median work better
        dists[dists == 0] = np.nan
        ranges[ranges == 0] = np.nan
        alts = zs.copy()
        alts[alts == 0] = np.nan

        #smoothed_dist = np.empty(alts.shape, dtype=np.float32)
        #smoothed_dist[:] = np.nan
        #for ind, row in enumerate(dists):
        #    avg_dist = np.nanmedian(ranges[ind, :])
        #    if np.isnan(avg_dist):
        #        continue
        #    window_size = 0.5 / (avg_dist * azi_delta)
        #    smoothed_dist[ind, :] = ndimage.median_filter(row, np.maximum(1, int(window_size)))

        #    #median_filter doesn't handle nans properly
        #    nans = np.isnan(row)
        #    nans_d = cv2.dilate(nans.astype(np.uint8), np.ones(int(window_size)))
        #    smoothed_dist[ind, nans_d[:, 0]>0] = np.nan
        smoothed_dist = dists

        smoothed_vert_range = np.cos(elevs)[:, None] * smoothed_dist
        smoothed_vert_alt = np.abs(np.sin(elevs)[:, None]) * smoothed_dist

        noise = 0.05
        max_slope = 0.25

        #horizontal deriv
        alt_delta = np.empty(alts.shape, dtype=np.float32)
        alt_delta[:] = np.nan
        for ind, row in enumerate(smoothed_vert_alt):
            avg_dist = np.nanmedian(ranges[ind, :])
            if np.isnan(avg_dist):
                continue
            window_size = 0.5 / (avg_dist * azi_delta)
            ws_n = int(np.maximum(window_size/2, 1))
            alt_delta[ind, :] = np.maximum(np.abs(np.roll(smoothed_vert_alt[ind, :], -ws_n) - np.roll(smoothed_vert_alt[ind, :], ws_n)) - noise*np.abs(np.sin(elevs[ind, None])), 0) / (azi_delta * smoothed_vert_range[ind, :] * (ws_n*2+1))
        #alt_delta = np.maximum(np.abs(np.roll(smoothed_vert_alt, -2, axis=1) - np.roll(smoothed_vert_alt, 2, axis=1)) - noise*np.abs(np.sin(elevs[:, None])), 0) / (azi_delta * smoothed_vert_range * 5)

        #find short vertical obstacles
        #alt_delta_vert = np.maximum(np.abs((smoothed_vert_alt - np.roll(smoothed_vert_alt, 1, axis=0))) - noise*np.abs(np.sin(elevs[:, None])), 0) / \
        #        (np.abs(smoothed_vert_range - np.roll(smoothed_vert_range, 1, axis=0)) + noise*np.abs(np.cos(elevs[:, None])))
        alt_delta_vert = np.zeros(alts.shape, dtype=np.float32)
        #find shallower obstacles
        for row_ind in reversed(range(smoothed_vert_alt.shape[0])):
            # LiDAR ~0.4m above ground
            lidar_h = 0.4
            pred_range = lidar_h / np.tan(-elevs[row_ind])
            delta = 1
            new_pred_range = lidar_h / np.tan(-elevs[row_ind - delta])
            while new_pred_range - pred_range < 0.5 and row_ind - (delta + 1) > 0 and new_pred_range > 0:
                delta += 1
                new_pred_range = lidar_h / np.tan(-elevs[row_ind - delta])

            alt_delta_vert[row_ind, :] += np.maximum(np.abs((smoothed_vert_alt[row_ind, :] - smoothed_vert_alt[row_ind-delta, :])) - noise*np.abs(np.sin(elevs[row_ind, None])), 0) / \
                    (np.abs(smoothed_vert_range[row_ind, :] - smoothed_vert_range[row_ind-delta, :]) + noise*np.abs(np.cos(elevs[row_ind, None])))

        obs = np.sqrt(alt_delta**2 + alt_delta_vert**2) > max_slope
        obs = np.nan_to_num(obs)
        obs = morphology.remove_small_objects(obs, 10)

        obs_inf = np.zeros(obs.shape, dtype=np.float32)
        #inflate along altitude
        for col_ind, col in enumerate(obs.T):
            initial_range = np.inf
            for row_ind, pt in enumerate(col):
                if pt > 0 and ranges[row_ind, col_ind] < initial_range:
                    initial_range = ranges[row_ind, col_ind]
                if initial_range != 0 and np.abs(initial_range - ranges[row_ind, col_ind]) < 0.5:
                    obs_inf[row_ind, col_ind] = 1

        #inflate along azimuth
        for ind, row in enumerate(obs_inf):
            avg_dist = np.nanmedian(ranges[ind, :])
            if np.isnan(avg_dist):
                continue
            window_size = 0.5 / (avg_dist * azi_delta)
            obs_inf[ind, :] = cv2.dilate(row[None, :].astype(np.uint8), np.ones([1, int(window_size)]))

        obs_inf = np.maximum(obs.astype(np.float32)*3, obs_inf)

        #find safe area, expanding from home
        for col_ind, col in enumerate(obs_inf.T):
            last_range = 0
            for row_ind, pt in reversed(list(enumerate(col))):
                if not np.isnan(dists[row_ind, col_ind]):
                    dist = np.abs(last_range - ranges[row_ind, col_ind])
                    if last_range == 0 or (dist < 1 and pt == 0):
                        last_range = ranges[row_ind, col_ind]
                        obs_inf[row_ind, col_ind] = 2
                    elif last_range != 0:
                        break

        #visualize rough regions
        obs_inf[np.sqrt(alt_delta**2 + alt_delta_vert**2) > 0.15] = np.maximum(2.5, obs_inf[np.sqrt(alt_delta**2 + alt_delta_vert**2) > 0.15])

        #VISUALIZATION
        #ROS viz
        viz_msg = PointCloud2()
        viz_msg.header.frame_id = "map"

        pos_field_x = PointField()
        pos_field_x.name = 'x'
        pos_field_x.offset = 0
        pos_field_x.datatype = PointField.FLOAT32
        pos_field_x.count = 1
        pos_field_y = PointField()
        pos_field_y.name = 'y'
        pos_field_y.offset = 4
        pos_field_y.datatype = PointField.FLOAT32
        pos_field_y.count = 1
        pos_field_z = PointField()
        pos_field_z.name = 'z'
        pos_field_z.offset = 8
        pos_field_z.datatype = PointField.FLOAT32
        pos_field_z.count = 1
        pos_field_label = PointField()
        pos_field_data = PointField()
        pos_field_data.name = 'data'
        pos_field_data.offset = 12
        pos_field_data.datatype = PointField.FLOAT32
        pos_field_data.count = 1

        viz_msg.fields.append(pos_field_x)
        viz_msg.fields.append(pos_field_y)
        viz_msg.fields.append(pos_field_z)
        viz_msg.fields.append(pos_field_data)

        viz_msg.point_step = 16
        viz_msg.height = 1
        viz_msg.width = 0
        viz_msg.width = xs.size
        
        stacked_data = np.vstack([xs.flatten(), -ys.flatten(), zs.flatten(), obs_inf.flatten()]).astype(np.float32).T
        viz_msg.data = stacked_data.tobytes()
        self.viz_pub_.publish(viz_msg)

        #pc_points = Points([xs.flatten(), ys.flatten(), zs.flatten()]).cmap("rainbow", obs_inf.flatten())
        #show(pc_points, resetcam=False, camera={'pos':(0,0,30)}).close()

        #self.viz_img(dists, 'dists')
        #self.viz_img(smoothed_dist, 'dists_smoothed')
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
    pp.parse(start_ind = 0)
