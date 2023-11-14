import cv2 as cv
import numpy as np


class KCF:
    def __init__(self):
        self.trackers = []

    def __call__(self, image, detections):
        # Propagate existing tracks
        tracks = np.empty(shape=(0, 6))
        for track in self.trackers:
            ok, (x, y, w, h) = track.update(image)
            trk = np.array([x, y, x+w, y+h, 0, 0])
            tracks = np.vstack((tracks, trk))

        self.trackers = []
        for detection in detections:
            tracker = cv.TrackerKCF_create()
            bbox = np.array([detection[0], detection[1], detection[2]-detection[0], detection[3]-detection[1]]).astype(int)
            ok = tracker.init(image, bbox)
            self.trackers.append(tracker)

        return tracks

    def create_track(self, image, detection):
        tracker = cv.TrackerKCF_create()
        bbox = np.array([detection[0], detection[1], detection[2]-detection[0], detection[3]-detection[1]]).astype(int)
        ok = tracker.init(image, bbox)
        self.trackers.append(tracker)

    def __repr__(self):
        return 'KCF'
