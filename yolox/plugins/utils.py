import os
import sys
import random
from pathlib import Path

import numpy as np
import cv2
import cv2 as cv
import torch
import torchvision.transforms.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment

from maskrcnn_benchmark.structures.bounding_box import BoxList

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def min_cost_matching(detections, bboxes, iou_thres):
    if len(bboxes) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections), dtype=int), np.empty(len(bboxes), dtype=int)
    iou_matrix = iou_batch(detections[:, :4], bboxes[:, :4])
    iou_matrix[np.isnan(iou_matrix)] = 0

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_thres).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            assign = np.stack(np.where(a), axis=1)
        else:
            x, y = linear_assignment(1-iou_matrix)
            assign = np.array(list(zip(x, y)))
    else:
        assign = np.empty(shape=(0, 2))

    unmatched_dets = [idx for idx, det in enumerate(detections) if idx not in assign[:, 0]]
    unmatched_bboxes = [idx for idx, bbox in enumerate(bboxes) if idx not in assign[:, 1]]

    matched = np.empty((0, 2), dtype=int)
    for m in assign:
        if (iou_matrix[m[0], m[1]] < iou_thres):
            unmatched_dets.append(m[0])
            unmatched_bboxes.append(m[1])
        else:
            matched = np.vstack((matched, m))

    return matched, np.array(unmatched_dets, dtype=int), np.array(unmatched_bboxes, dtype=int)


def transform_detections(detections, rot_mat):
    num_detections = len(detections)
    corner_points = np.ones((4 * num_detections, 3))
    corner_points[:, :2] = detections[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * num_detections, 2)
    corner_points = corner_points @ rot_mat.T
    corner_points = corner_points.reshape(num_detections, 8)
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]

    bboxes = np.concatenate((corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))).reshape(4, num_detections).T
    detections[:, :4] = bboxes
    return detections

def postprocess_detections_to_ndarray(prediction, img_size, confthre=0.3):
    boxes = [] # for each image
    for idx, pred in enumerate(prediction):
        if pred is None:
            boxes.append(np.empty(shape=(0, 4)))
            continue
        pred = pred[pred[:, 4] * pred[:, 5] >= confthre, :]
        pred = pred[pred[:, 2] - pred[:, 0] < 0.3 * img_size[0], :]
        pred = pred[pred[:, 3] - pred[:, 1] < 0.3 * img_size[1], :]
        boxes.append(pred.cpu().numpy())
    return boxes


def postprocess_detections_to_boxlist(prediction, img_size, confthre=0.3):
    boxes = []
    for idx, pred in enumerate(prediction):
        if pred is None:
            box = BoxList(torch.empty((0, 4), dtype=torch.float32, device='cuda:0'), img_size, mode='xyxy')
            box.add_field('ids', torch.tensor([], dtype=torch.int64, device='cuda:0'))
            box.add_field('labels', torch.tensor([], dtype=torch.int64, device='cuda:0'))
            box.add_field('objectness', torch.tensor([], dtype=torch.float32, device='cuda:0'))
            box.add_field('scores', torch.tensor([], dtype=torch.float32, device='cuda:0'))
            boxes.append(box)
            continue
        pred = pred[pred[:, 4] * pred[:, 5] >= confthre, :]
        pred = pred[pred[:, 2] - pred[:, 0] < 0.3 * img_size[0], :]
        pred = pred[pred[:, 3] - pred[:, 1] < 0.3 * img_size[1], :]
        box = BoxList(pred[:, :4], img_size, mode='xyxy')
        box.add_field('ids', torch.tensor([-1 for i in range(len(pred))], dtype=torch.int64, device=pred.device))
        box.add_field('labels', pred[:, -1].clone())
        box.add_field('objectness', pred[:, 4].clone())
        box.add_field('scores', pred[:, 5].clone())
        boxes.append(box)
    return boxes


def postprocess_tracks_from_ndarray(tracks, img_size, info=None):
    boxes = [] # for each image
    for idx, (track, orig_h, orig_w) in enumerate(zip(tracks, info[0], info[1])):
        if len(track) == 0:
            boxes.append(np.empty((0, 7)))
            continue
        scale = min(img_size[0] / float(orig_h), img_size[1] / float(orig_w))
        track[:, :4] = track[:, :4] / scale
        boxes.append(track)
    return boxes


def postprocess_tracks_from_boxlist(tracks, img_size, info=None):
    boxes = []
    for idx, (boxlist, orig_h, orig_w) in enumerate(zip(tracks, info[0], info[1])):
        if len(boxlist) == 0:
            boxes.append(torch.empty((0, 7)))
            continue
        scale = min(img_size[0] / float(orig_h), img_size[1] / float(orig_w))
        objectness = boxlist.get_field('objectness')[None, ].T
        scores = boxlist.get_field('scores')[None, ].T
        labels = boxlist.get_field('labels')[None, ].T
        dets = torch.cat((boxlist.bbox / scale, objectness, scores, labels), dim=1)
        boxes.append(dets)
    return boxes


def postprocess_images(imgs):
    imgs = torch.cat([img[[2,1,0], :, :] for img in imgs]).unsqueeze(0)
    imgs = F.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return imgs


def draw_bbox(img, bbox, color=None, label=None, line_thickness=None):
    tl = (line_thickness or round(0.002 * (img.shape[0]+img.shape[1])/2) + 1)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if not label:
        return
    tf = max(tl, 1)  # font thickness
    text_size = cv2.getTextSize(label, 0, fontScale=tl / 2, thickness=tf)[0]
    c2 = c1[0] + text_size[0], c1[1] - text_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        img, label, (c1[0], c1[1] - 2), 0, tl / 2,
        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,
    )


