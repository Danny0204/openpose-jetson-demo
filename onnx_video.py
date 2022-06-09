import math
import time
from operator import itemgetter
import cv2
import onnxruntime as ort
import numpy as np
import threading
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose


def masking_and_show(resized, outputs):
    img = resized.copy()
    num_keypoints = Pose.num_kpts

    heatmaps = outputs[0]
    heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

    pafs = outputs[1]
    pafs = np.transpose(pafs.squeeze(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    for pose in current_poses:
        pose.draw(img)
    img = cv2.addWeighted(resized, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                      (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
    cv2.imshow('Lightweight Human Pose Estimation Python Demo', cv2.resize(img, (512, 512)))

    if cv2.waitKey(1) == ord('q'):
        return


if __name__ == '__main__':
    ort_sess = ort.InferenceSession('pose_256.onnx',
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        resized = cv2.resize(frame, (256, 256))
        image = resized.copy()
        image = image.astype("float32").transpose(2, 0, 1)
        image = (image - 128) / 256
        image = np.expand_dims(image, axis=0)
        start_time = time.time()
        outputs = ort_sess.run(None, {'image': image})
        print('inference time: %s sec' % (time.time() - start_time))
        threading.Thread(target=masking_and_show, args=(resized, outputs)).run()

    cap.release()
    cv2.destroyAllWindows()
