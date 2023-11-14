import pdb

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from .kcf import KCF
from .kf import KF
from .horizon import HorizonDetector
from .utils import transform_detections, iou_batch, \
    postprocess_detections_to_ndarray, \
    postprocess_detections_to_boxlist, \
    postprocess_tracks_from_ndarray, \
    postprocess_tracks_from_boxlist, \
    postprocess_images
from .clustering import Clustering

class Plugins:
    def __init__(self, args, exp):
        self.test_size = exp.test_size
        self.fusionthre = args.fusionthre
        self.trackthre = args.trackthre
        self.orig_size = (3000, 4096)

        self.hzdet = None
        if args.with_hzdet:
            self.hzdet = HorizonDetector(*self.orig_size, args.horizon_ckpt)

        self.clust = None
        if args.with_dbscan:
            self.clust = Clustering(exp.config, eps=args.eps, minpts=args.minpts)

        self.tracker = None
        if args.with_siamese:
            from siammot.modelling.rcnn import build_siammot
            from siammot.configs.defaults import cfg
            cfg.merge_from_file(args.tracker_config)
            self.tracker = build_siammot(cfg)
            self.tracker.cuda()
            self.tracker.eval()
            ckpt = torch.load(args.tracker_ckpt, map_location='cuda')
            self.tracker.load_state_dict(ckpt['model'])
        elif args.with_kf:
            self.tracker = KF()
        elif args.with_kcf:
            self.tracker = KCF()

    def filter_detections_by_horizon(self, outputs, horizons):
        if self.hzdet is None:
            return
        for idx, (output, horizon) in enumerate(zip(outputs, horizons)):
            if output is None: continue
            xmax, ymax = output[:, 2].cpu().numpy(), output[:, 3].cpu().numpy()
            x1, y1, x2, y2 = horizon.flatten()
            h = ((y2-y1)/(x2-x1)) * xmax + y1
            outputs[idx] = output[ymax > h - 100, :]

    def horizon(self, imgs):
        if self.hzdet is None:
            return imgs, [None for _ in imgs], [None for _ in imgs], [None for _ in imgs], [None for _ in imgs]
        ret = [self.hzdet(img) for img in imgs] # (img, horizon, rotation)
        return list(zip(*ret))

    def track(self, imgs, orig_imgs, outputs, info, aux_dict=None):
        if self.tracker is None:
            return [np.empty((0, 7)) for _ in imgs]
        if isinstance(self.tracker, KCF):
            detections = postprocess_detections_to_ndarray(outputs, self.test_size, confthre=self.trackthre)
            tracks = [self.tracker(img.cpu().numpy(), det) for img, det in zip(orig_imgs, detections)]
            tracks = postprocess_tracks_from_ndarray(tracks, self.test_size, info=info)
        elif isinstance(self.tracker, KF):
            detections = postprocess_detections_to_ndarray(outputs, self.test_size, confthre=self.trackthre)
            tracks = [self.tracker(det) for det in detections]
            tracks = postprocess_tracks_from_ndarray(tracks, self.test_size, info=info)
        else:
            detections = postprocess_detections_to_boxlist(outputs, self.test_size, self.trackthre)
            imgs = postprocess_images(imgs)
            tracks = self.tracker(imgs, given_detection=detections, info=info)
            tracks = postprocess_tracks_from_boxlist(tracks, self.test_size, info=info)
            tracks = [(track.cpu().numpy()) if track is not None else None for track in tracks]
        return tracks

    def clustering(self, ids, rots=None):
        if self.clust is None:
            return [np.empty((0, 4)) for _ in ids]
        fmt = '{}/PointcloudsNP/{}.npy'
        clusters = [self.clust(np.load(fmt.format(img_id[0], img_id[1][:14] + img_id[1][-6:])), img_id[1][14])[0]
                    for img_id in ids]
        if rots:
            clusters = [transform_detections(cluster, rot) if rot is not None else cluster
                        for cluster, rot in zip(clusters, rots)]
        return clusters

    def decide(self, results, clusters_list, tracks_list, uuids=None, aux_dict=None):
        if self.clust is None and self.tracker is None:
            return results

        outputs = {}
        for key, value, clusters, tracks, uuid in zip(
                results.keys(), results.values(), clusters_list, tracks_list, uuids
        ):
            bboxes, classes, scores = value
            if bboxes is None:
                detections = np.empty(shape=(0, 6))
            else:
                bboxes, classes, scores = [v.cpu().numpy() for v in value]
                detections = np.hstack((bboxes, classes[None, ].T, scores[None, ].T))

            cond = detections[:, 5] >= self.fusionthre

            clusters_mask = np.zeros(len(detections), dtype=bool)
            if clusters is not None:
                clusters_mask = self.merge_clusters(detections, clusters)

            tracks_mask = np.zeros(len(detections), dtype=bool)
            if tracks is not None:
                tracks_mask = self.merge_tracks(detections, tracks)

            tracks_clusters_mask = np.zeros(len(tracks), dtype=bool)
            if tracks_clusters_mask.any():
                print(tracks_clusters_mask)

            detections = detections[cond | (clusters_mask & (~cond)) | (tracks_mask & (~cond))]

            bboxes = torch.from_numpy(detections[:, :4])
            classes = torch.from_numpy(detections[:, 4])
            scores = torch.from_numpy(detections[:, 5])
            outputs[key] = [bboxes, classes, scores]

            aux_dict[uuid].update({
                'clusters_mask': clusters_mask,
                'tracks_mask': tracks_mask,
                'tracks_clusters_mask': tracks_clusters_mask,
            })

        return outputs

    def merge_tracks_and_clusters(self, tracks, clusters):
        mask = np.zeros(len(tracks), dtype=bool)
        matched, unmatched_trks, unmatched_clsts = self.match_detections(tracks, clusters, iou_thres=0.3)
        mask[matched[:, 0]] = True
        return mask

    def merge_clusters(self, detections, clusters):
        mask = np.zeros(len(detections), dtype=bool)
        cond = detections[:, 5] >= self.fusionthre
        highconf, lowconf = detections[cond], detections[(~cond)]
        where_highconf, where_lowconf = np.where(cond)[0], np.where(~cond)[0]
        first_matched, unmatched_dets, unmatched_clsts = self.match_detections(highconf, clusters, iou_thres=0.1)
        mask[where_highconf[first_matched[:, 0]]] = True
        clusters = clusters[unmatched_clsts]
        second_matched, unmatched_dets, unmatched_clsts = self.match_detections(lowconf, clusters, iou_thres=0.3)
        mask[where_lowconf[second_matched[:, 0]]] = True
        return mask

    def merge_tracks(self, detections, tracks):
        mask = np.zeros(len(detections), dtype=bool)
        cond = detections[:, 5] >= self.fusionthre
        highconf, lowconf = detections[cond], detections[~cond]
        where_highconf, where_lowconf = np.where(cond)[0], np.where(~cond)[0]
        first_matched, unmatched_dets, unmatched_trks = self.match_detections(highconf, tracks, iou_thres=0.1)
        mask[where_highconf[first_matched[:, 0]]] = True
        tracks = tracks[unmatched_trks]
        second_matched, unmatched_dets, unmatched_trks = self.match_detections(lowconf, tracks, iou_thres=0.4)
        mask[where_lowconf[second_matched[:, 0]]] = True
        return mask

    def match_detections(self, detections, bboxes, iou_thres=0.5):
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

    def reverse(self, results, info, rots):
        if self.hzdet is None:
            return

        for key, val, height, width, rot in zip(results.keys(), results.values(), *info, rots):
            bboxes, classes, scores = val
            if bboxes is None: continue
            rev = np.insert(rot, 2, values=np.array([0, 0, 1]), axis=0)
            rev = np.linalg.inv(rev)[:2, :]
            bboxes = torch.from_numpy(transform_detections(bboxes.cpu().numpy(), rev))
            results[key] = bboxes

    def __repr__(self):
        return 'hzdet={}, clust={}, tracker={}'.format(
            self.hzdet, self.clust, type(self.tracker),
        )


def build_plugins(args, exp):
    return Plugins(args, exp)
