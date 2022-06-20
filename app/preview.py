import queue
import threading as mt
import time

import cv2
import numpy as np

from app.animation import Animation
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose


class PreviewWindow:
    def __init__(self, src_weight: float = 0.6, pose_weight: float = 0.4):
        self.window_name = 'OpenPose'
        self.src_weight = src_weight
        self.pose_weight = pose_weight
        self.pose_score_threshold = 10.0

        self._queue = queue.Queue()
        self._stop_event = mt.Event()
        self._thread = mt.Thread(target=self.window_worker, args=(self.src_weight, self.pose_weight), daemon=True)

        self.is_capturing = False
        self._buffer = None

    def show_image_with_heatmaps(self, image, heatmaps, pafs, upsampling_ratio, resize_ratio):
        self._queue.put((image, heatmaps, pafs, upsampling_ratio, resize_ratio))

    def show_image_with_pose(self, image: np.ndarray, pose: Pose):
        self._queue.put((image, pose))

    def _extract_poses(self, heatmaps, pafs, upsampling_ratio, resize_ratio):
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
            if pose_entries[n][18] < self.pose_score_threshold:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        # print('postprocess time:', time.time() - _t2)
        return current_poses

    def _show_image_with_pose(self, image, pose, src_weight, pose_weight):
        if self.is_capturing:
            self._buffer.insert(round(time.time() * 1000), pose)
        _t4 = time.time()
        canvas = image.copy()
        if pose is not None:
            pose.draw(canvas)
            cv2.rectangle(canvas, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        img_to_show = cv2.addWeighted(image, src_weight, canvas, pose_weight, 0)
        # img_to_show = cv2.resize(img_to_show, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # print('draw time:', time.time() - _t4)
        cv2.imshow(self.window_name, img_to_show)
        key = cv2.waitKey(10)
        if key == ord('c'):
            self.is_capturing = not self.is_capturing
            if self.is_capturing:
                self._buffer = Animation(image.shape)
                print('capture start')
            else:
                filename = 'capture_%d.oanim' % round(time.time())
                self._buffer.export(filename)
                print('capture file saved as', filename)
                self._buffer = None

    def window_worker(self, src_weight, pose_weight):
        while True:
            if self._stop_event.is_set():
                break
            result = self._queue.get()
            if len(result) == 2:
                image, pose = result
                self._show_image_with_pose(image, pose, src_weight, pose_weight)
            elif len(result) == 5:
                image, heatmaps, pafs, upsampling_ratio, resize_ratio = result
                poses = self._extract_poses(heatmaps, pafs, upsampling_ratio, resize_ratio)
                pose = poses[0] if len(poses) > 0 else None
                self._show_image_with_pose(image, pose, src_weight, pose_weight)
            self._queue.task_done()

    def start(self):
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
