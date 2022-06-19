import struct
from typing import List, Tuple

from modules.pose import Pose


class Animation:
    poses: List[Pose]

    def __init__(self, image_size: Tuple[int, int], fps: int):
        self.image_size = image_size
        self.fps = fps
        self.poses = []

    def insert(self, pose: Pose):
        self.poses.append(pose)

    def length(self):
        return 1.0 / self.fps * len(self.poses)

    def export(self, path: str):
        with open(path, 'wb') as f:
            f.write(struct.pack('QQQI', self.image_size[0], self.image_size[1], len(self.poses), self.fps))
            for p in self.poses:
                for kpt in p.keypoints:
                    f.write(struct.pack('BB', kpt[0], kpt[1]))
