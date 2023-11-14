#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import pdb
import copy
import os
import sys
import tempfile
import time
from collections import ChainMap
from loguru import logger
from tqdm import tqdm

import numpy as np

import torch
import torchvision.transforms.functional as F

from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized


def preproc(imgs, input_size, device):
    imgs = torch.permute(imgs, (0, 3, 1, 2))
    scale = min(input_size[0] / imgs.shape[2], input_size[1] / imgs.shape[3])
    resized_imgs = F.resize(imgs, (int(imgs.shape[2] * scale), int(imgs.shape[3] * scale)))
    padded_imgs = torch.ones((imgs.shape[0], imgs.shape[1], input_size[0], input_size[1]), dtype=torch.uint8, device=device) * 114
    padded_imgs[:, :, :resized_imgs.shape[2], :resized_imgs.shape[3]] = resized_imgs
    padded_imgs = padded_imgs.contiguous()
    return padded_imgs

class VOCEvaluator:
    """
    VOC AP Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        exp_dir=None,
        part=None,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        # self.dataloader.dataset.preproc = None
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)

        self.exp_dir = exp_dir
        self.part = part
        self.params = {}

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        plugins=None,
        visual=False,
    ):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        total_time = 0
        track_time = 0
        preproc_time = 0
        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        self.aux_dict = {} # {image_uuid: {}}
        if plugins is not None and plugins.hzdet is not None:
            self.dataloader.dataset.preproc = None
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            # skip the last iters since batchsize might be not enough for batch inference
            is_time_record = cur_iter < len(self.dataloader) - 1
            if is_time_record:
                start = time.time()

            uuids = [self.dataloader.dataset.ids[i][1] for i in ids]
            self.aux_dict.update({uuid: {} for uuid in uuids})
            # sea-sky line detection
            if plugins is not None and plugins.hzdet is not None:
                rot_imgs, horizons_orig, rots_orig, horizons, rots = plugins.horizon(imgs.numpy())
                imgs = torch.stack([torch.from_numpy(img) for img in rot_imgs])
                orig_imgs = imgs
                imgs = preproc(imgs, test_size, device='cuda')
            else:
                orig_imgs = imgs
                horizons = [None for _ in imgs]
                horizons_orig = [None for _ in imgs]
                rots = [None for _ in imgs]
                rots_orig = [None for _ in imgs]

            imgs = imgs.type(tensor_type)
            if is_time_record:
                preproc_end = time_synchronized()
                preproc_time += preproc_end - start

            with torch.no_grad():
                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - preproc_end

                # [x1, y1, x2, y2, obj conf, cls conf, cls]
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

                # if plugins is not None:
                #     plugins.filter_detections_by_horizon(outputs, horizons)

                # tracking
                if plugins is not None:
                    tracks = plugins.track(imgs, orig_imgs, outputs, info_imgs, aux_dict=self.aux_dict)

                if is_time_record:
                    track_end = time_synchronized()
                    track_time += track_end - nms_end

            results = self.convert_to_voc_format(outputs, info_imgs, ids)


            # clustering
            if plugins is not None:
                image_ids = [self.dataloader.dataset.ids[i] for i in ids]
                clusters = plugins.clustering(image_ids, rots_orig)

            # decision strategy
            if plugins is not None:
                results = plugins.decide(results, clusters, tracks, uuids=uuids, aux_dict=self.aux_dict)

            data_list.update(results)

            if plugins is not None:
                for uuid, t, c, rot, hz_orig, rot_orig, result, img in zip(
                    uuids, tracks, clusters, rots, horizons_orig, rots_orig, results.values(), imgs,
                ):
                    self.aux_dict[uuid].update({
                        'tracks': t, 'clusters': c,
                        'rotation': rot,
                        'horizon_orig': hz_orig, 'rotation_orig': rot_orig,
                        'orig': result,
                        'input': img,
                    })

            if is_time_record:
                total_end = time_synchronized()
                total_time += total_end - start

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        self.params.update({
            'conf': self.confthre,
            'nms': self.nmsthre,
            'fusionthre': plugins.fusionthre if plugins is not None else 0.0,
            'horizon_on': bool(plugins and plugins.horizon),
            'track_on': bool(plugins and plugins.tracker),
            'clust_on': bool(plugins and plugins.clust),
            'part': self.dataloader.dataset.image_set[0][1],
            'eval_summary_path': os.path.join(self.exp_dir, 'results.csv'),
            'tracker': type(plugins.tracker) if plugins and plugins.tracker else None,
        })
        total_samples = n_samples * self.dataloader.batch_size
        for k, v in zip(['infer', 'track', 'preproc', 'total'],
                        [inference_time, track_time, preproc_time, total_time]):
            data = {
                '{}_time'.format(k): '{:.4f} s'.format(v),
                '{}_speed'.format(k): '{:.4f} fps'.format(total_samples/v),
                'avg_{}_time'.format(k): '{:.4f} ms'.format(1000*v/total_samples),
            }
            self.params.update(data)

        eval_results = self.evaluate_prediction(data_list, statistics)
        if visual:
            self.dataloader.dataset.visualize_detections(self.exp_dir, data_list, self.aux_dict, self.params)
        synchronize()
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        all_boxes = [
            [[] for _ in range(self.num_images)] for _ in range(self.num_classes)
        ]
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images)
            )
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.dataloader.dataset.evaluate_detections(
                all_boxes, tempdir, exp_dir=self.exp_dir, aux_dict=self.aux_dict, params=self.params,
            )
            return mAP50, mAP70, info
