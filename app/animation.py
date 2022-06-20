import struct
from typing import Tuple

from modules.pose import Pose


class Animation:
    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size
        self.poses = []

    def insert(self, time: int, pose: Pose):
        self.poses.append((time, pose))

    def length(self):
        return self.poses[-1][0] - self.poses[0][0]

    def export(self, path: str):
        with open(path, 'wb') as f:
            f.write(struct.pack('iiq', self.image_size[0], self.image_size[1], len(self.poses)))
            for p in self.poses:
                f.write(struct.pack('q', p[0] - self.poses[0][0]))
                for kpt in p[1].keypoints:
                    f.write(struct.pack('ii', kpt[0], kpt[1]))
