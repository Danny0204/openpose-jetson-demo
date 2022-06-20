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

        self.input_size = (256, 256)
        self.upsampling_ratio = 2
        self.pose_score_threshold = pose_score_threshold

    def inference(self, image: np.ndarray):
        _t1 = time.time()
        outputs = self.ort_session.run(None, {'image': image})

        heatmaps = outputs[0]
        heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsampling_ratio, fy=self.upsampling_ratio,
                              interpolation=cv2.INTER_CUBIC)

        pafs = outputs[1]
        pafs = np.transpose(pafs.squeeze(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsampling_ratio, fy=self.upsampling_ratio,
                          interpolation=cv2.INTER_CUBIC)
        # print('inference time: ', time.time() - _t1)

        return heatmaps, pafs

    def inference_with_poses(self, image: np.ndarray):
        resize_ratio = (image.shape[1] / 256, image.shape[0] / 256)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = image.astype("float32").transpose(2, 0, 1)
        image = (image - 128) / 256
        image = np.expand_dims(image, axis=0)

        heatmaps, pafs = self.inference(image)

        _t2 = time.time()
        num_keypoints = Pose.num_kpts
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * 8 / self.upsampling_ratio * resize_ratio[0])
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * 8 / self.upsampling_ratio * resize_ratio[1])
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
