import os
import sys
import random

import yaml
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN


def read_config(path):
    """Read yaml config file to dict"""
    with open(path, 'r', encoding='utf-8') as fin:
        config = yaml.load(fin.read(), Loader=yaml.FullLoader)
    return config


def drop_points_behind_camera(pts, K, D, R, t):
    """
    pts: numpy array of [x, y, z, ...]
    K: intrinsic matrix
    D: distort matrix
    R: rotation matrix
    t: translation vector
    """
    xyz = pts[:, :3]
    proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
    # convert 3D points into homgenous points
    xyz_hom = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xy_hom = np.dot(proj_mat, xyz_hom.T).T
    # get 2d coordinates in image [pixels]
    z = xy_hom[:, -1]
    xy = xy_hom[:, :2] / np.tile(z[:, np.newaxis], (1, 2))
    # undistort - has to be 1xNx2 structure
    xy = cv.undistortPoints(np.expand_dims(xy, axis=0), np.eye(3), D).squeeze()
    # drop all points behind camera
    pts = pts[z > 5]
    return pts


class Clustering:
    def __init__(self, config, eps, minpts):
        self.config = read_config(config)
        self.height, self.width = self.config['height'], self.config['width']
        self.eps = eps
        self.minpts = minpts

    def __repr__(self):
        return 'Clustering(eps={}, minpts={}, height={}, width={})'.format(
            self.eps, self.minpts, self.height, self.width,
        )

    def preprocess(self, a):
        return a[a[:, 1] > 5, :]

    def __call__(self, pts, camera_id):
        rvec = np.array(self.config['rvec'][camera_id])
        tvec = np.array(self.config['tvec'][camera_id])
        intrinsic = np.array(self.config['camera_matrix'][camera_id])
        distort = np.array(self.config['camera_distort'][camera_id])

        detections = np.empty(shape=(0, 5))

        # Filtering
        pts = self.preprocess(pts)
        if pts.shape[0] == 0:
            return detections, np.empty(shape=(0, 7))
        pts = drop_points_behind_camera(pts, intrinsic, distort, cv.Rodrigues(rvec)[0], tvec)
        if pts.shape[0] == 0:
            return detections, np.empty(shape=(0, 7))

        # Projecting: data format is [u, v, x, y, z, d]
        proj, _ = cv.projectPoints(pts.astype(np.float32), rvec, tvec, intrinsic, distort)
        proj = proj.reshape(-1, 2)
        depth = np.linalg.norm(pts[:, :3], 2, axis=1)

        # Clustering
        pts = pts[(proj[:, 0] >= 0) & (proj[:, 0] < self.width) & (proj[:, 1] >= 0) & (proj[:, 1] < self.height)]
        depth = depth[(proj[:, 0] >= 0) & (proj[:, 0] < self.width) & (proj[:, 1] >= 0) & (proj[:, 1] < self.height)]
        proj = proj[(proj[:, 0] >= 0) & (proj[:, 0] < self.width) & (proj[:, 1] >= 0) & (proj[:, 1] < self.height)]
        if pts.shape[0] == 0:
            return detections, np.empty(shape=(0, 7))

        db = DBSCAN(eps=self.eps, min_samples=self.minpts).fit(pts)
        labels = db.labels_.reshape(-1, 1)
        num_labels = len(set(list(db.labels_)))

        data = np.hstack((proj, pts, labels, depth[np.newaxis].T))

        for i in range(num_labels):
            obj = data[data[:, 5] == i, :]
            if obj.shape[0] == 0: continue
            depth = np.median(obj[:, 6])
            xmin, ymin = np.min(obj[:, 0:2], axis=0).astype(int)
            xmax, ymax = np.max(obj[:, 0:2], axis=0).astype(int)
            detections = np.vstack((detections, (xmin, ymin, xmax, ymax, depth)))
        for idx, detection in enumerate(detections):
            if detection[2] - detection[0] < 5:
                detections[idx][0] -= 100
                detections[idx][2] += 100
            if detection[3] - detection[1] < 5:
                detections[idx][1] -= 50
                detections[idx][3] += 50
        return detections, data
