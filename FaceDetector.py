import threading
import time
from typing import Any, List, Optional, Tuple
from video_source import VideoSource        # replaces: from camera_source import CameraSource

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from features import WindowFeatureExtractor, WindowFeatures, FrameSignals, build_frame_signals
from camera_source import CameraSource

# Each landmark is a MediaPipe NormalizedLandmark (has .x, .y, .z as floats in 0-1).
# I use Any for the element type to avoid depending on MediaPipe's internal import path.
Landmarks = List[Any]


class FaceDetector:
    """Runs MediaPipe Face Landmarker on the webcam in a background thread
    and exposes the latest annotated frame and eye aspect ratio (EAR)."""

    # --- Class attributes (shared by all instances, treated as constants) ---
    MODEL_PATH: str = "face_landmarker.task"
    # FaceMesh indices: [corner, upper, upper, corner, lower, lower]
    RIGHT_EYE: List[int] = [33, 160, 158, 133, 153, 144]
    LEFT_EYE: List[int] = [362, 385, 387, 263, 373, 380]

    def __init__(self, source: VideoSource) -> None:   # was: source: CameraSource
        self.source: VideoSource = source
        """Load the model and set up the thread state. Does not open the camera."""
        # Settings for the detector
        self.options: vision.FaceLandmarkerOptions = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,  # sequence of frames, enables tracking
            num_faces=1,                            # only the driver
            output_facial_transformation_matrixes=True,
        )
        # The detector itself (loading the model is slow, so we do it once here)
        self.landmarker: vision.FaceLandmarker = vision.FaceLandmarker.create_from_options(self.options)

        # --- Thread control ---
        self._stop_event: threading.Event = threading.Event()   # set() tells the loop to exit
        self._thread: Optional[threading.Thread] = None        # None until start() is called
        self._lock: threading.Lock = threading.Lock()          # protects the two fields below

        # --- Latest results, written by the thread, read by get_latest() ---
        self.latest_frame: Optional[np.ndarray] = None  # annotated BGR image, None before the first frame
        self.latest_ear: Optional[float] = None         # None means no face detected

        self.extractor: WindowFeatureExtractor = WindowFeatureExtractor()
        self.latest_features: Optional[WindowFeatures] = None

    def start(self) -> None:
        """Start the background thread. Returns immediately (does not block)."""
        if self._thread and self._thread.is_alive():
            return  # already running, do nothing
        self._stop_event.clear()  # reset the flag in case we are restarting
        self._thread = threading.Thread(target=self._loop, daemon=True)  # daemon: dies with the program
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to exit and wait until it has finished."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()  # blocks until _loop returns

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[WindowFeatures]]:
        """returns: (frame, ear, features); any of them can be None"""
        with self._lock:
            return self.latest_frame, self.latest_ear, self.latest_features
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
            h, w = frame.shape[:2]  # frame size in pixels

            # OpenCV uses BGR, MediaPipe expects RGB
            rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # MediaPipe VIDEO mode needs strictly increasing timestamps in ms (capture time of the frame)
            timestamp_ms: int = int(t * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw every landmark as a small green dot (display only)
            if result.face_landmarks:  # empty list means no face
                points: Landmarks = result.face_landmarks[0]  # first (only) face
                for p in points:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)

            # Per-frame measurements, then the rolling window features
            signals: FrameSignals = build_frame_signals(t, result, w, h)
            self.extractor.add(signals)
            features: Optional[WindowFeatures] = self.extractor.compute()

            # Publish the results so the main thread can read them
            with self._lock:
                self.latest_frame = frame
                self.latest_ear = signals.ear  # None if no face
                self.latest_features = features