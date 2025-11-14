import time
import cv2
import numpy as np
import os
from dataclasses import dataclass
from ktamv_server_io import Ktamv_Server_Io as io

IMG_W, IMG_H = 640, 480
CENTER = np.array([IMG_W//2, IMG_H//2])
MAX_FAIL = 3
FRAME_RESET = 100

SAVE_ROOT_DIR = "/tmp/nozzle_detection_results"
call_count = 0

@dataclass
class NozzleAlgo:
    pre_idx: int
    detector: cv2.SimpleBlobDetector
    color: tuple
    aid: int

CFG_12 = [
    (0, 'standard', (0, 0, 255), 1),
    (1, 'standard', (0, 255, 0), 2),
    (2, 'standard', (39, 255, 127), 3),
    (3, 'standard', (255, 0, 255), 4),
    (0, 'relaxed',  (255, 0, 0), 5),
    (1, 'relaxed',  (39, 127, 255), 6),
    (2, 'relaxed',  (39, 255, 127), 7),
    (3, 'relaxed',  (0, 255, 255), 8),
]

class Ktamv_Server_Detection_Manager:
    uv = [None, None]
    __algorithm = None
    __io = None
    def __init__(self, log, camera_url, cloud_url, send_to_cloud=False, *a, **kw):
        self.log = log
        self.send_to_cloud = send_to_cloud
        self.__io = io(log=log, camera_url=camera_url, cloud_url=cloud_url, save_image=False)

        self._base_params = self._setup_base_params()
        self._algos = self._build_algorithms()
        self._success = [0]*(len(CFG_12)+1)
        self._fail_cnt=0
        self._last_size=None
        self._frame_cnt=0
        self._init_save_root()

    def _init_save_root(self):
        if not os.path.exists(SAVE_ROOT_DIR):
            os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
            self.log(f"Created image save root directory: {SAVE_ROOT_DIR}")

        else:
            self.log(f"Image save root directory already exists: {SAVE_ROOT_DIR}")

    def _create_call_dir(self):
        global call_count
        call_count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        call_dir = os.path.join(SAVE_ROOT_DIR, f"call_{call_count}_{timestamp}")
        os.makedirs(call_dir, exist_ok=True)
        self.log(f"Created image save folder for this call: {call_dir}")
        return call_dir

    def _save_drawn_image(self, call_dir, drawn_img, frame_idx):
        img_filename = f"frame_{frame_idx}_{time.strftime('%Y%m%d_%H%M%S_%f', time.localtime())[:-3]}.png"
        img_save_path = os.path.join(call_dir, img_filename)
        cv2.imwrite(img_save_path, drawn_img) 
        self.log(f"Image saved to: {img_save_path}")

    def recursively_find_nozzle_position(self, put_frame_func, min_matches, timeout, xy_tol):
        current_call_dir = self._create_call_dir()
        current_frame_idx = 0

        start, counter, last = time.time(), {}, None
        while time.time() - start < timeout:
            frame = self.__io.get_single_frame()
            pos, vis = self.nozzleDetection(frame)

            if vis is not None:
                self._save_drawn_image(current_call_dir, vis, current_frame_idx)
                current_frame_idx += 1
                put_frame_func(vis)
            
            if pos is None:
                continue
            key = (int(pos[0]), int(pos[1]))
            counter[key] = counter.get(key, 0) + 1
            last = pos
            if counter[key] >= min_matches:
                if self.send_to_cloud:
                    self.__io.send_frame_to_cloud(frame, pos, self.__algorithm)
                break
            time.sleep(0.3)
        
        self.log(f"Call completed. A total of {current_frame_idx} images saved to: {current_call_dir}")
        return last

    def get_preview_frame(self, put_frame_func):
        _, vis = self.nozzleDetection(self.__io.get_single_frame())
        if vis is not None:
            put_frame_func(vis)

    def nozzleDetection(self, img):
        if img is None:
            return None, None
        center = self._detect_blob(img)
        vis = self._draw(img.copy(), center)
        return center, vis

    def _setup_base_params(self):
        return {
            'standard': {
                'minArea': 2000, 'maxArea': 5000,
                'minCircularity': 0.6, 'minConvexity': 0.8,
                'filterByArea': True, 'filterByCircularity': True,
                'filterByConvexity': True
            },
            'relaxed': {
                'minArea': 1000, 'maxArea': 7000,
                'minCircularity': 0.4, 'minConvexity': 0.6,
                'filterByArea': True, 'filterByCircularity': True,
                'filterByConvexity': True
            },
        }

    def _build_algorithms(self):
        def make(pkey):
            p = cv2.SimpleBlobDetector_Params()
            src = self._base_params[pkey]
            for k, v in src.items():
                setattr(p, k, v)
            return cv2.SimpleBlobDetector_create(p)

        return [NozzleAlgo(pre, make(key), color, aid)
                for pre, key, color, aid in CFG_12]

    def _preprocess(self, img, idx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if idx == 0:
            y = cv2.GaussianBlur(gray, (5, 5), 3)
            return cv2.adaptiveThreshold(y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 2)
        if idx == 1:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            return cv2.GaussianBlur(th, (5, 5), 3)
        if idx == 2:
            return cv2.medianBlur(gray, 3)
        return gray

    def _detect_blob(self, img):
        if self.__algorithm:
            algo = self._algos[self.__algorithm-1]
            pt = self._try_algo(img, algo)
            if pt:
                self._success[algo.aid] += 1
                self._fail_cnt = 0
                return pt
            self._fail_cnt += 1
            if self._fail_cnt >= MAX_FAIL:
                self.__algorithm = None

        for algo in sorted(self._algos, key=lambda x: self._success[x.aid], reverse=True):
            pt = self._try_algo(img, algo)
            if pt:
                self.__algorithm = algo.aid
                self._fail_cnt = 0
                return pt
        return None

    def _try_algo(self, img, algo):
        kps = algo.detector.detect(self._preprocess(img, algo.pre_idx))
        if not kps:
            return None
        kp = min(kps, key=lambda k: np.linalg.norm(np.array(k.pt) - CENTER))
        x, y, s = kp.pt[0], kp.pt[1], kp.size
        if not (20 <= x <= IMG_W-20 and 20 <= y <= IMG_H-20):
            return None
        if self._last_size and not (0.5 <= s / self._last_size <= 2):
            return None
        self._last_size = s
        return int(round(x)), int(round(y))

    def _draw(self, img, center):
        cx, cy = IMG_W//2, IMG_H//2
        if center:
            cv2.circle(img, center, int(self._last_size//2), (0, 255, 0), -1)
            cv2.line(img, (center[0]-5, center[1]), (center[0]+5, center[1]), (255, 255, 255), 2)
            cv2.line(img, (center[0], center[1]-5), (center[0], center[1]+5), (255, 255, 255), 2)
        else:
            r = 17
            cv2.circle(img, (cx, cy), r, (0, 0, 0), 3)
            cv2.circle(img, (cx, cy), r+1, (0, 0, 255), 1)
        cv2.line(img, (cx, 0), (cx, IMG_H), (0, 0, 0), 2)
        cv2.line(img, (0, cy), (IMG_W, cy), (0, 0, 0), 2)
        cv2.line(img, (cx, 0), (cx, IMG_H), (255, 255, 255), 1)
        cv2.line(img, (0, cy), (IMG_W, cy), (255, 255, 255), 1)
        return img