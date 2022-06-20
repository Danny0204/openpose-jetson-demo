import queue
import threading as mt
import time

import numpy as np
import cv2

from typing import Tuple

from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose

pose_score_threshold = 10.0


def __extract_poses(heatmaps, pafs, upsampling_ratio, resize_ratio):
    _t2 = time.time()
    num_keypoints = Pose.num_kpts
    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * 8 / upsampling_ratio * resize_ratio[0])
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * 8 / upsampling_ratio * resize_ratio[1])
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        if pose_entries[n][18] < pose_score_threshold:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
    print('postprocess time: ', time.time() - _t2)
    return current_poses


def __show_image_with_pose(image, pose, src_weight, pose_weight, window_size, window_title):
    _t4 = time.time()
    canvas = image.copy()
    if pose is not None:
        pose.draw(canvas)
        cv2.rectangle(canvas, (pose.bbox[0], pose.bbox[1]),
                      (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
    img_to_show = cv2.addWeighted(image, src_weight, canvas, pose_weight, 0)
    img_to_show = cv2.resize(img_to_show, window_size)
    print('draw time: ', time.time() - _t4)
    cv2.imshow(window_title, img_to_show)
    cv2.waitKey(1)


def window_worker(queue, stop_event, src_weight, pose_weight, window_size, window_title):
    while True:
        if stop_event.is_set():
            break
        result = queue.get()
        if len(result) == 2:
            image, pose = result
            __show_image_with_pose(image, pose, src_weight, pose_weight, window_size, window_title)
        elif len(result) == 5:
            image, heatmaps, pafs, upsampling_ratio, resize_ratio = result
            poses = __extract_poses(heatmaps, pafs, upsampling_ratio, resize_ratio)
            pose = poses[0] if len(poses) > 0 else None
            __show_image_with_pose(image, pose, src_weight, pose_weight, window_size, window_title)
        queue.task_done()


class PreviewWindow:
    def __init__(self, window_size: Tuple[int, int] = (1024, 768),
                 window_title: str = 'OpenPose 2D Motion Capture',
                 src_weight: float = 0.6,
                 pose_weight: float = 0.4):
        self.window_size = window_size
        self.window_title = window_title
        self.src_weight = src_weight
        self.pose_weight = pose_weight

        self._queue = queue.Queue()
        self._stop_event = mt.Event()
        self._thread = mt.Thread(target=window_worker,
                                 args=(self._queue, self._stop_event, self.src_weight, self.pose_weight,
                                       self.window_size, self.window_title),
                                 daemon=True)

    def show_image_with_heatmaps(self, image, heatmaps, pafs, upsampling_ratio, resize_ratio):
        self._queue.put((image, heatmaps, pafs, upsampling_ratio, resize_ratio))

    def show_image_with_pose(self, image: np.ndarray, pose: Pose):
        self._queue.put((image, pose))

    def start(self):
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
