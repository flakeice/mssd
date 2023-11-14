"""
Implements the Generalized R-CNN for SiamMOT
"""
import pdb
import torch
from torch import nn
from loguru import logger

import siammot.operator_patch.run_operator_patch

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from .backbone.backbone_ext import build_backbone
from .track_head.track_head import build_track_head
from .track_head.track_utils import build_track_utils
from .track_head.track_solver import build_track_solver


class SiamMOT(nn.Module):
    """
    Main class for R-CNN. Currently supports boxes and tracks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and
             computes detections / tracks from it.
    """

    def __init__(self, cfg):
        super(SiamMOT, self).__init__()

        self.backbone = build_backbone(cfg)

        track_utils, track_pool = build_track_utils(cfg)
        self.track = build_track_head(cfg, track_utils, track_pool)
        self.solver = build_track_solver(cfg, track_pool)

    def reset_siammot_status(self):
        self.track.reset_track_pool()

    def forward(self, images, targets=None, given_detection=None, info=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if self.training:
            proposals = postprocess_proposals(targets)
        else:
            proposals = None

        losses = {}
        y, tracks, loss_track = self.track(features, proposals, targets)
        losses.update(loss_track)

        # solver is only needed during inference
        if not self.training:
            detections = given_detection
            scores = detections[0].get_field('scores')
            # detections[0].add_field('scores', scores - 0.01)

            # visualize_inference(images.tensors, tracks, detections, info=info, track_memory=self.track.track_memory)

            # if tracks is not None:
            #     tracks[0].add_field('objectness', torch.tensor([1.0 for _ in tracks[0].bbox], device=tracks[0].bbox.device))
            #     detections = [cat_boxlist(detections + tracks)]

            new_tracks = self.solver(detections)
            self.track.flush_track_memory(features, new_tracks)
            tracks = get_dummy_boxlist(detections[0].size) if tracks is None else tracks

        if self.training:
            return tracks, losses
        else:
            return tracks


def postprocess_proposals(targets):
    # for training
    proposals = copy.deepcopy(targets)
    boxes = []
    for i, box in enumerate(proposals):
        deivce = box.bbox.device
        box.extra_fields.pop('ids')
        box.add_field('ids', torch.tensor([-1 for _ in range(len(box))], dtype=torch.int64, device=device))
        box.add_field('objectness', torch.tensor([0.99 for _ in range(len(box))], dtype=torch.float64, device=device))
        boxes.append(box)
    return boxes


def get_dummy_boxlist(image_size):
    dummy_boxlist = BoxList(torch.empty((0, 4), device='cuda:0'), image_size, mode='xyxy')
    dummy_boxlist.add_field('ids', torch.tensor([], device='cuda:0'))
    dummy_boxlist.add_field('labels', torch.tensor([], device='cuda:0'))
    dummy_boxlist.add_field('scores', torch.tensor([], device='cuda:0'))
    dummy_boxlist.add_field('objectness', torch.tensor([], device='cuda:0'))
    return [dummy_boxlist]


def build_siammot(cfg):
    siammot = SiamMOT(cfg)
    return siammot
