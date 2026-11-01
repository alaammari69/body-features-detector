import threading
import time
from typing import Optional, Tuple
from video_source import VideoSource, open_source   # replaces: from camera_source import CameraSource
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from body_features import (BodySignals, BodyWindowFeatureExtractor, BodyWindowFeatures,
                           build_body_signals, Landmarks)


class BodyDetector:
    """Runs MediaPipe Pose Landmarker on the webcam in a background thread
    and exposes the latest annotated frame, per-frame signals and window features."""

    # --- Class attributes (shared by all instances, treated as constants) ---
    MODEL_PATH: str = "pose_landmarker_full.task"
    # Landmarks drawn on screen: nose, shoulders, elbows, wrists, hips
    DRAW_IDX: Tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16, 23, 24)

    def __init__(self, source: VideoSource) -> None:   # was: source: CameraSource
        self.source: VideoSource = source
        """Load the model and set up the thread state. Does not open the camera."""

        self.options: vision.PoseLandmarkerOptions = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,  # sequence of frames, enables tracking
            num_poses=1,                            # only the driver
        )
        self.landmarker: vision.PoseLandmarker = vision.PoseLandmarker.create_from_options(self.options)
        self.extractor: BodyWindowFeatureExtractor = BodyWindowFeatureExtractor()

        # --- Thread control ---
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()  # protects the three fields below

        # --- Latest results, written by the thread, read by get_latest() ---
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_signals: Optional[BodySignals] = None
        self.latest_features: Optional[BodyWindowFeatures] = None

    def start(self) -> None:
        """Start the background thread. Returns immediately (does not block)."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to exit and wait until it has finished."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[BodySignals], Optional[BodyWindowFeatures]]:
        """Thread-safe read of the newest results.

        returns: (frame, signals, features); any of them can be None
                 signals.pose_found tells you if a usable body was seen in the latest frame
        """
        with self._lock:
            return self.latest_frame, self.latest_signals, self.latest_features

    def _loop(self) -> None:
        """The detection loop. Runs in the background thread (internal use)."""
        last_id: int = -1  # id of the last frame we processed
        while not self._stop_event.is_set():
            frame: Optional[np.ndarray]
            frame_id: int
            t: float
            frame, frame_id, t = self.source.get(last_id)  # frame is None if there is no NEW frame yet

            if frame is None:
                if not self.source.is_running():
                    print("Camera stopped")
                    break
                time.sleep(0.001)  # nothing new yet, don't burn CPU
                continue
            last_id = frame_id

            h: int
            w: int
            h, w = frame.shape[:2]

            rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms: int = int(t * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw the key body points and the shoulder line (display only)
            if result.pose_landmarks:
                points: Landmarks = result.pose_landmarks[0]
                for i in self.DRAW_IDX:
                    cv2.circle(frame, (int(points[i].x * w), int(points[i].y * h)), 5, (0, 255, 0), -1)
                cv2.line(frame,
                         (int(points[11].x * w), int(points[11].y * h)),
                         (int(points[12].x * w), int(points[12].y * h)), (255, 0, 0), 2)

            # Per-frame measurements, then the rolling window features
            signals: BodySignals = build_body_signals(t, result, w, h)
            self.extractor.add(signals)
            features: Optional[BodyWindowFeatures] = self.extractor.compute()

            # Publish the results so the main thread can read them
            with self._lock:
                self.latest_frame = frame
                self.latest_signals = signals
                self.latest_features = features