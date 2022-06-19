import time

import cv2
import numpy as np
import onnxruntime as ort

from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose


class PoseNetONNX:
    def __init__(self, onnx_file: str, ort_providers=None, pose_score_threshold: float = 10.0):
        if ort_providers is None:
            ort_providers = ['CPUExecutionProvider']

        self.onnx_file = onnx_file
        self.ort_providers = ort_providers
        self.ort_session = ort.InferenceSession(onnx_file, providers=ort_providers)
        self.pose_score_threshold = pose_score_threshold

    def inference(self, image: np.ndarray):
        _t1 = time.time()
        outputs = self.ort_session.run(None, {'image': image})
        print(f'ort run time: {time.time() - _t1}')

        heatmaps = outputs[0]
        heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

        pafs = outputs[1]
        pafs = np.transpose(pafs.squeeze(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs

    def inference_pose(self, image: np.ndarray):
        num_keypoints = Pose.num_kpts
        heatmaps, pafs = self.inference(image)

        _t2 = time.time()
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)
        # print(f'kpt time: {time.time() - _t2}')

        _t3 = time.time()
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    kpt = all_keypoints[int(pose_entries[n][kpt_id])]
                    pose_keypoints[kpt_id, 0] = int(kpt[0])
                    pose_keypoints[kpt_id, 1] = int(kpt[1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            if pose.confidence < self.pose_score_threshold:
                continue
            current_poses.append(pose)
        # print(f'pose time: {time.time() - _t2}')

        return current_poses
