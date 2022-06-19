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
        frame = cv2.resize(frame, (256, 256))
        frame_t = frame.copy().astype("float32").transpose(2, 0, 1)
        frame_t = (frame_t - 128) / 256
        frame_t = np.expand_dims(frame_t, axis=0)

        poses = net.inference_pose(frame_t)

        window.show_image_with_pose(frame, poses[0] if len(poses) > 0 else None)
