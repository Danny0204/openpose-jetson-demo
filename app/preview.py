import threading
import time

import numpy as np
import cv2

from typing import Tuple
from queue import Queue

from modules.pose import Pose


class PreviewWindow:
    def __init__(self, window_size: Tuple[int, int] = (1024, 768),
                 window_title: str = 'OpenPose 2D Motion Capture',
                 src_weight: float = 0.6,
                 pose_weight: float = 0.4):
        self.window_size = window_size
        self.window_title = window_title
        self.src_weight = src_weight
        self.pose_weight = pose_weight

        self._queue = Queue()
        self._thread = threading.Thread(target=self._window_worker, daemon=True)
        self._stop_event = threading.Event()

    def _show_image_with_pose(self, image: np.ndarray, pose: Pose):
        canvas = image.copy()
        if pose is not None:
            pose.draw(canvas)
            cv2.rectangle(canvas, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        img_to_show = cv2.addWeighted(image, self.src_weight, canvas, self.pose_weight, 0)
        img_to_show = cv2.resize(img_to_show, self.window_size)
        cv2.imshow(self.window_title, img_to_show)
        cv2.waitKey(1)

    def _window_worker(self):
        while True:
            if self._stop_event.is_set():
                break
            image, pose = self._queue.get()
            self._show_image_with_pose(image, pose)
            self._queue.task_done()

    def show_image_with_pose(self, image: np.ndarray, pose: Pose):
        self._queue.put((image, pose))

    def start(self):
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        if self._thread.is_alive():
            self._stop_event.set()
