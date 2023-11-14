import sys
import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment as linear_assignment
from filterpy.kalman import KalmanFilter
from loguru import logger

from .utils import min_cost_matching

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    if x[2] * x[3] < 0:
        logger.warning('invalid value encountered in sqrt: {} {}'.format(x[2], x[3]))
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])
        self.kf.P *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.Q[4:, 4:] *= 0.5
        self.kf.Q[6:,6:] *= 0.5
        self.kf.R[2:, 2:] *= 10.  # 观测值的方差记为10个像素
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.miss_count = 0
        self.hit_streak = 0

    def update(self, bbox=None):
        if bbox is None:
            self.predict()
            self.miss_count += 1
            self.hit_streak = 0
        else:
            self.kf.update(convert_bbox_to_z(bbox))
            self.miss_count = 0
            self.hit_streak += 1

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return convert_x_to_bbox(self.kf.x_prior)

    def state(self):
        return convert_x_to_bbox(self.kf.x)


class KF:
    def __init__(self):
        self.tracks = []

    def __call__(self, detections):
        """
        detections: format is (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        """
        # Predict existing tracks
        tracks = np.empty(shape=(0, 6))
        for track in self.tracks:
            track.predict()
            state = track.state().flatten()
            if np.any(np.isnan(state)):
                continue
            box = np.array([*state[0:4].tolist(), 0, 0])
            tracks = np.vstack((tracks, box))

        # Create, delete or update tracks using detections
        matched, unmatched_dets, unmatched_trks = min_cost_matching(detections, tracks, iou_thres=0.4)
        for (det_idx, trk_idx) in matched:
            self.update_track(trk_idx, detections[det_idx][:4])
        for trk_idx in unmatched_trks:
            self.update_track(trk_idx)
        for det_idx in unmatched_dets:
            self.create_track(detections[det_idx][:4])
        for track in self.tracks:
            if track.miss_count >= 1: self.tracks.remove(track)
        return tracks

    def create_track(self, detection):
        self.tracks.append(KalmanTracker(detection))

    def update_track(self, trk_idx, detection=None):
        self.tracks[trk_idx].update(detection)

    def __repr__(self):
        return 'KF'
