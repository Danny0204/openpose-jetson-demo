import time

import cv2
import numpy as np

from app.net import PoseNetONNX
from app.preview import PreviewWindow

if __name__ == '__main__':
    net = PoseNetONNX(onnx_file='pose_256.onnx',
                      ort_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    window = PreviewWindow()
    window.start()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        resize_ratio = (frame.shape[1] / 256, frame.shape[0] / 256)
        frame_r = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frame_r = frame_r.astype("float32").transpose(2, 0, 1)
        frame_r = (frame_r - 128) / 256
        frame_r = np.expand_dims(frame_r, axis=0)

        _to = time.time()
        heatmaps, pafs = net.inference(frame_r)
        print('overall time: ', time.time() - _to)

        window.show_image_with_heatmaps(frame, heatmaps, pafs, net.upsampling_ratio, resize_ratio)
