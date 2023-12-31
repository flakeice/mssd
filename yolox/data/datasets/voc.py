#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET
from loguru import logger

import torch
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
from torchvision.utils import save_image


from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
from .voc_classes import VOC_CLASSES


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True, class_names=VOC_CLASSES):
        self.class_to_ind = class_to_ind or dict(
            zip(class_names, range(len(class_names)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None and difficult.text is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info


class VOCDetection(Dataset):

    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        image_sets=[("2007", "trainval"), ("2012", "trainval")],
        img_size=(416, 416),
        preproc=None,
        target_transform=None,
        dataset_name="VOC0712",
        cache=False,
        class_names=VOC_CLASSES,
    ):
        super().__init__(img_size)
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        if target_transform is None:
            self.target_transform = AnnotationTransform(class_names=class_names)
        else:
            self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self.class_names = class_names
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            for line in open(
                os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            ):
                self.ids.append((rootpath, line.strip()))

        self.annotations = self._load_coco_annotations()
        self.imgs = None
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.root, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {self._imgpath % img_id} not found"

        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        return img, target, img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None, exp_dir=None, aux_dict=None, params=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou, exp_dir=exp_dir, aux_dict=aux_dict, params=params)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.class_names):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5, exp_dir=None, aux_dict=None, params=None):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(
            self.root, "annotations_cache", "VOC" + self._year, name
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.class_names):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap, tp, fp, npos = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
                aux_dict=aux_dict,
                params=params,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            precision = prec[-1]
            recall = rec[-1]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0.0
            print('Precision = {}'.format(precision))
            print('Recall = {}'.format(recall))
            print('F1 score = {}'.format(f1))
            print('TP = {}, FP = {}, FN = {}'.format(tp[-1], fp[-1], npos-tp[-1]))
            print("Mean AP = {:.4f}".format(np.mean(aps)))

            summary = {
                'part': self.image_set[0][1], 'iou': iou,
                'tp': tp[-1], 'fp': fp[-1], 'fn': npos-tp[-1],
                'prec': precision, 'rec': recall, 'f1': f1,
                'mAP': float(np.mean(aps)),
            }
            summary.update(params)
            df = pd.DataFrame([pd.Series(summary)])
            outpath = params['eval_summary_path']
            df.to_csv(outpath, mode='a', index=False, header=not os.path.exists(outpath))

        return np.mean(aps)

    def visualize_detections(self, exp_dir, data_list, aux_dict=None, params=None):
        print('Visualizing...')
        import pandas as pd
        import shutil
        from pathlib import Path
        from tqdm import tqdm
        from yolox.plugins.utils import draw_bbox
        from yolox.plugins.kcf import KCF
        from yolox.plugins.kf import KF
        GREEN_COLOR = (0, 255, 0)
        DET_LOW_COLOR = (0, 0, 255)
        DET_HIGH_COLOR = (255, 0, 0)
        DETECTION_COLOR = tuple(reversed([0, 113, 188]))
        TRACK_COLOR = tuple(reversed([247, 147, 30]))
        CLUSTER_COLOR = tuple(reversed([0, 146, 69]))
        HORIZON_COLOR = tuple(reversed([211, 54, 130]))
        KCF_COLOR = tuple(reversed([105, 102, 178]))
        KF_COLOR = tuple(reversed([255, 121, 198]))
        ANNO_COLOR = tuple(reversed([102, 255, 0]))
        if params['tracker'] == KF:
            TRACK_COLOR = KF_COLOR
        elif params['tracker'] == KCF:
            TRACK_COLOR = KCF_COLOR

        out_dir = Path(exp_dir) / 'vis'
        os.makedirs(out_dir, exist_ok=True)
        if params['tracker'] == KF or params['tracker'] == KCF:
            pass
        else:
            for index, uuid in enumerate(tqdm(self.ids)):
                filepath = Path(self._imgpath % uuid)
                shutil.copyfile(filepath, Path(out_dir) / filepath.name, follow_symlinks=True)

        # Draw detection results
        print('Drawing detection...')
        for (key, val), aux in zip(data_list.items(), tqdm(aux_dict.values())):
            uuid = self.ids[key]
            bboxes, classes, scores = val
            if bool(aux) is False:
                outpath = out_dir / '{}.jpg'.format(uuid[1])
                img = cv2.imread(str(outpath), cv2.IMREAD_COLOR)
                if bboxes is not None:
                    for idx, (bbox, cls, score) in enumerate(zip(bboxes, classes, scores)):
                        draw_bbox(img, bbox, color=DETECTION_COLOR, label='{:.3f}'.format(score.item()), line_thickness=4)
                cv.imwrite(str(outpath), img)
                continue

            outpath = out_dir / '{}.jpg'.format(uuid[1])
            img = cv2.imread(str(outpath), cv2.IMREAD_COLOR)

            rotation, horizon = aux['rotation_orig'], aux['horizon_orig']
            p1, p2 = horizon.astype(int).tolist()
            img = cv.line(img, p1, p2, HORIZON_COLOR, thickness=6)
            if params['tracker'] == KF or params['tracker'] == KCF:
                rot_img = img
            else:
                rot_img = cv.warpAffine(img, rotation, (img.shape[1], img.shape[0]))

            try:
                tracks = aux['tracks']
                tracks_mask = aux['tracks_mask']
            except KeyError as e:
                tracks = []
                tracks_mask = np.zeros(len(bboxes) if bboxes is not None else 0, dtype=bool)
            try:
                clusters = aux['clusters']
                clusters_mask = aux['clusters_mask']
            except KeyError as e:
                clusters = []
                clusters_mask = np.zeros(len(bboxes) if bboxes is not None else 0, dtype=bool)

            if params['tracker'] == KF or params['tracker'] == KCF:
                pass
            else:
                bboxes, classes, scores = aux['orig']
                if bboxes is not None:
                    for idx, (bbox, cls, score, track_mask, cluster_mask) in enumerate(zip(
                            bboxes, classes, scores, tracks_mask, clusters_mask,
                    )):
                        text = ' {}{}'.format('T' if track_mask else '', 'C' if cluster_mask else '')
                        draw_bbox(rot_img, bbox, color=DETECTION_COLOR, label='{:.3f}{}'.format(score.item(), text), line_thickness=4)

            for idx, track in enumerate(tracks):
                draw_bbox(rot_img, track[:4], color=TRACK_COLOR, line_thickness=4)
            for idx, cluster in enumerate(clusters):
                draw_bbox(rot_img, cluster[:4], color=CLUSTER_COLOR, line_thickness=4)

            cv.imwrite(str(outpath), rot_img)
