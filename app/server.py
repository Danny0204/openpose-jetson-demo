import queue
import socket
import struct
import threading
from typing import List


class PoseServer:
    conn_list: List[socket.socket]

    def __init__(self, queue: queue.Queue):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', 23456))
        self.socket.listen(5)
        print('real time pose server running on port', 23456)
        self.conn_list = []
        self.queue = queue

    def handle_conn(self):
        while True:
            conn, addr = self.socket.accept()
            self.conn_list.append(conn)

    def handle_pose(self):
        while True:
            pose = self.queue.get()
            buffer = b''
            for kpt in pose.keypoints:
                buffer += struct.pack('ii', kpt[0], kpt[1])
            for conn in self.conn_list:
                try:
                    conn.sendall(buffer)
                except OSError:
                    conn.close()
                    self.conn_list.remove(conn)

    def start(self):
        threading.Thread(target=self.handle_conn, daemon=True).start()
        threading.Thread(target=self.handle_pose, daemon=True).start()
