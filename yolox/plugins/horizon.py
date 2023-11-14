import pdb
import os
import sys
import argparse
import math
import random
import time
import json
import csv
import logging
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PATCH_SIZE = 32


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(2592, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        output = self.out(x)
        return output, x


class HorizonDetector:
    def __init__(self, height, width, ckpt, rank=0):
        self.model = MyCNN().cuda(rank)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()

        self.rng = np.random.default_rng(42)
        self.height = height
        self.width = width
        self.lsd = cv.createLineSegmentDetector()

        self.std_width = 640
        self.std_height = int(self.std_width * (self.height / self.width))
        self.angle_threshold = 10.0
        self.center_scale = 0.5

        self.default_level = self.std_height * self.center_scale
        self.default_horizon = np.array([[0, self.default_level], [self.width, self.default_level]])
        self.prev = False
        self.prev_horizon = self.default_horizon
        self.prev_angle = 0.0
        self.prev_center = self.default_level
        self.first_center = None
        self.accum_horizon = []
        self.accum_angle = []
        self.accum_center = []

    def __repr__(self):
        return 'HorizonDetector(hw=({}, {}), std_hw=({}, {}), angle_threshold={}, center_scale={})'.format(
            self.height, self.width, self.std_height, self.std_width, self.angle_threshold, self.center_scale,
        )

    def __call__(self, img, info=None, scale=None):
        """Detect horizion in the given image
        img: OpenCV format image
        info: additional information about given image
        scale: for resizing the image
        return: horizon, rotation matrix
        """
        t1 = time.time()
        horizon = self._find_horizon(img, scale=self.std_width/img.shape[1]) # horizon in img shape
        horizon = self._fix_horizon(img, horizon)
        rot = self._calc_rotation(img, horizon)
        img = self._rotate(img, rot)
        orig_scale = img.shape[1]/self.width
        orig_horizon = horizon / orig_scale # horizon in orig shape
        orig_rot = self._calc_rotation_scale(img, orig_horizon, orig_scale)
        return img, orig_horizon, orig_rot, horizon, rot

    def _fix_horizon(self, img, horizon):
        fr, to = horizon[:, 1]
        center = [horizon[1, 0] / 2, (fr + to) / 2]
        slope = (horizon[1, 1] - horizon[0, 1]) / (horizon[1, 0] - horizon[0, 0])
        angle = np.degrees(np.arctan2(to - fr, horizon[1, 0]))
        if self.prev and ((np.abs(angle - self.prev_angle) > 3) or (np.abs(center[1] - self.prev_center) > img.shape[0]/25)):
            horizon = self.prev_horizon
        return horizon

    def _find_horizon(self, img, scale=None):
        img = img.copy()
        if scale and scale != 1.0:
            img = cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

        lines = self._detect_lines(img)
        if len(lines) == 0:
            return self.prev_horizon

        points, slopes = self._filter_lines(img, lines)
        if len(points) == 0:
            return self.prev_horizon

        points, slopes = self._filter_points(img, points, slopes)
        if len(points) < 2:
            return self.prev_horizon

        horizon = self._ransac(img, points, slopes)

        x1, y1 = 0.0, get_y(horizon, 0.0)
        x2, y2 = img.shape[1], get_y(horizon, img.shape[1])
        horizon = np.array([[x1, y1], [x2, y2]])

        if scale and scale != 1.0:
            horizon = horizon / scale

        return horizon

    def _detect_lines(self, img):
        dst_img = img.copy()
        dst_img = cv.cvtColor(dst_img, cv.COLOR_BGR2GRAY)
        sobel = cv.Sobel(dst_img, cv.CV_64F, 0, 1, ksize=1)
        sobel = np.uint8(np.absolute(sobel))
        dst_img = cv.addWeighted(dst_img, 1, sobel, 1, 1)
        lines = self.lsd.detect(dst_img)[0]
        return lines

    def _filter_lines(self, img, lines):
        points, slopes = [], []
        for line in lines:
            x1, y1, x2, y2 = np.array(line[0]).astype(np.float32)
            angle = np.arctan(float(y2 - y1) / float(x2 - x1)) * 180.0 / np.pi if x2 - x1 != 0.0 else 90.0
            slope = ((y2 - y1) / (x2 - x1)) if x2 - x1 != 0.0 else 1.0
            if math.fabs(angle) > self.angle_threshold:
                continue
            num_points = max(abs(x2 - x1) * 200 / img.shape[1], 0)
            if num_points < 10:
                continue
            delta_x = math.fabs(x2 - x1) / float(num_points)
            start_x = min(x1, x2)
            for i in range(int(num_points)):
                x = delta_x * float(i) + start_x
                y = (((y2 - y1) / (x2 - x1)) * (x - x1) + y1) if x2 - x1 != 0.0 else y1
                points.append([int(x), y])
                slopes.append(slope)
        return np.array(points), np.array(slopes)

    def _filter_points(self, img, points, slopes):
        patches, locs = get_patches(img, points)
        if len(patches) == 0:
            return np.array([]), np.array([])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        size = PATCH_SIZE
        patches = torch.from_numpy(patches)
        patches = np.transpose(patches, [0, 3, 1, 2]).float()
        patches = patches.to(device)
        preds, _ = self.model(patches)
        preds = preds.argmax(dim=1)
        filter_points = []
        filter_slopes = []
        for i in range(len(preds)):
            if preds[i] != 1: continue
            for j in range(len(points)):
                if (int(points[j][1] / size) == locs[i][0] and
                    int(points[j][0] / size) == locs[i][1]):
                    filter_points.append(points[j])
                    filter_slopes.append(slopes[j])
        return np.array(filter_points), np.array(filter_slopes)

    def _ransac(self, img, points, slopes, iters=200, sigma=3, slope_min=-1, slope_max=1):
        num_points = points.shape[0]
        line = [0, 0, 0, 0] # [x1, y1, x2, y2]
        best_score = 1
        most_num_points = 0
        for k in range(iters):
            idx1, idx2 = self.rng.choice(range(num_points), 2, replace=False)
            pt1, pt2 = points[idx1], points[idx2]
            dp = pt1 - pt2
            dp *= 1 / np.linalg.norm(dp)
            if dp[0] == 0: continue

            slope = dp[1] / dp[0]
            if slope_min <= slope and slope <= slope_max:
                vectors = points - pt1
                dist_losses = vectors[:, 1] * dp[0] - vectors[:, 0] * dp[1]
                slope_losses = slope - slopes

                # get satisfied points
                acc_points = points[(np.fabs(dist_losses) < sigma) & (np.fabs(slope_losses) < 0.1)][:, 0]
                score = len(acc_points)
                grid_acc_points = np.unique((acc_points / 4).astype(np.int32))
                num_acc_points = len(grid_acc_points)
                if ((score > best_score or num_acc_points > most_num_points)
                    and (score / best_score > 0.5
                         and num_acc_points / score > most_num_points / best_score
                         or score / best_score > 1.5)):
                    line = [dp[0], dp[1], pt1[0], pt1[1]]
                    best_score = score
                    most_num_points = num_acc_points
        return line

    def _calc_rotation_scale(self, img, horizon, scale):
        fr, to = horizon[:, 1]
        center = [horizon[1, 0] / 2, (fr + to) / 2]
        slope = (horizon[1, 1] - horizon[0, 1]) / (horizon[1, 0] - horizon[0, 0])
        angle = np.degrees(np.arctan2(to - fr, horizon[1, 0]))
        rot = cv.getRotationMatrix2D(center, angle, 1)
        rot[1][2] += self.first_center / scale - center[1]
        return rot

    def _calc_rotation(self, img, horizon, orig_scale=None):
        fr, to = horizon[:, 1]
        center = [horizon[1, 0] / 2, (fr + to) / 2]
        slope = (horizon[1, 1] - horizon[0, 1]) / (horizon[1, 0] - horizon[0, 0])
        angle = np.degrees(np.arctan2(to - fr, horizon[1, 0]))
        rot = cv.getRotationMatrix2D(center, angle, 1)

        if self.first_center is None:
            self.first_center = center[1]
        rot[1][2] += (self.first_center - center[1]) # offset to specific point

        self.accum_horizon.append(horizon)
        self.accum_angle.append(angle)
        self.accum_center.append(center)
        self.prev = True
        self.prev_horizon = horizon
        self.prev_angle = angle
        self.prev_center = center[1]
        return rot

    def _rotate(self, img, rot):
        img = cv.warpAffine(img, rot, (img.shape[1], img.shape[0]))
        return img


def get_y(line, x):
    if line[0] == 0:
        line[0] = max(0.00001, line[0])
    k = line[1] / line[0]
    return k * (x - line[2]) + line[3]


def get_patches(img, points):
    width, height, size = img.shape[1], img.shape[0], PATCH_SIZE
    num_width, num_height = math.ceil(width / size), math.ceil(height / size)
    patch_map = np.zeros((num_height, num_width))
    for point in points:
        patch_map[int(point[1] / size)][int(point[0] / size)] += 1
    labels = [[i, j] for i in range(num_height) for j in range(num_width)
              if patch_map[i][j] > 3]
    patches = []
    for i in range(len(labels)):
        x, y = labels[i]
        patch = cv.resize(img[x*size:x*size+size, y*size:y*size+size], (size, size))
        patches.append(patch)
    patches = np.array(patches)
    labels = np.array(labels)
    return patches, labels
