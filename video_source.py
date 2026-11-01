import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoSource(ABC):
    """Base class for anything that produces frames (camera, video file, ...).
    Grabs frames in a background thread and keeps only the newest one, so several
    detectors can read the same source. Subclasses only implement _open() and _next()."""

    def __init__(self) -> None:
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()  # protects the three fields below

        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0         # increases by 1 for every new frame
        self._frame_time: float = 0.0   # timestamp of the frame in seconds

    # ---------------- public API (same for every source) ----------------

    def start(self) -> None:
        """Start the capture thread. Returns immediately."""
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

    def is_running(self) -> bool:
        """False once the thread has ended (camera failed, video ended, or stop() was called)."""
        return self._thread is not None and self._thread.is_alive()

    def get(self, last_id: int = -1) -> Tuple[Optional[np.ndarray], int, float]:
        """Newest frame, but only if it is newer than `last_id`.

        returns: (frame, frame_id, time_s)
                 frame is None if there is no new frame yet. Otherwise it is a private COPY,
                 so each detector can draw on it without affecting the others.
        """
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, self._frame_id, self._frame_time
            return self._frame.copy(), self._frame_id, self._frame_time

    # ---------------- what a subclass must provide ----------------

    @abstractmethod
    def _open(self) -> Optional[cv2.VideoCapture]:
        """Open and configure the underlying capture. Return None if that failed."""

    @abstractmethod
    def _next(self, cap: cv2.VideoCapture) -> Optional[Tuple[np.ndarray, float]]:
        """Return (frame, time_s), or None when there are no more frames
        (camera failure or end of video). Timestamps must keep increasing."""

    # ---------------- shared loop ----------------

    def _loop(self) -> None:
        """Capture loop. Runs in the background thread (internal use)."""
        cap: Optional[cv2.VideoCapture] = self._open()
        if cap is None:
            return
        try:
            while not self._stop_event.is_set():
                item: Optional[Tuple[np.ndarray, float]] = self._next(cap)
                if item is None:
                    break
                frame: np.ndarray
                t: float
                frame, t = item
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._frame_time = t
        finally:
            cap.release()


class CameraSource(VideoSource):
    """A live webcam. Timestamps are wall-clock seconds since the camera opened."""

    def __init__(self, cam_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        super().__init__()
        self.cam_index: int = cam_index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps
        self._start: float = 0.0

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap: cv2.VideoCapture = cv2.VideoCapture(self.cam_index)  # if no image appears, try 1
        if not cap.isOpened():
            print("Could not open camera")
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # set first: allows HD at 30 fps over USB
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # The camera silently picks the nearest mode it supports, so print what we really got
        print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
              f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        self._start = time.time()
        return cap

    def _next(self, cap: cv2.VideoCapture) -> Optional[Tuple[np.ndarray, float]]:
        ok: bool
        frame: np.ndarray
        ok, frame = cap.read()
        if not ok:
            print("Could not read from camera")
            return None
        return frame, time.time() - self._start


class VideoFileSource(VideoSource):
    """A recorded video file. Timestamps come from the VIDEO's own timeline
    (frame number / fps), not from the clock, so the features are computed on the
    video's real timing no matter how fast or slow your computer processes it."""

    def __init__(self, path: str, realtime: bool = True, loop: bool = False) -> None:
        """
        path:     video file (mp4, avi, mov, ...)
        realtime: True = play at the video's own speed, like a live camera
                  False = read as fast as possible (see the note about skipped frames)
        loop:     True = restart from the beginning when the video ends
        """
        super().__init__()
        self.path: str = path
        self.realtime: bool = realtime
        self.loop: bool = loop
        self._fps: float = 30.0         # replaced by the file's real fps in _open()
        self._index: int = 0            # frames read in the current pass through the file
        self._offset: float = 0.0       # time added after each loop so timestamps keep increasing
        self._wall_start: float = 0.0   # when playback started (for realtime pacing)

    def _open(self) -> Optional[cv2.VideoCapture]:
        if not os.path.isfile(self.path):
            print(f"Video file not found: {self.path}")
            return None
        cap: cv2.VideoCapture = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"Could not open video: {self.path}")
            return None
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1.0 else 30.0  # some files report 0 or nonsense
        print(f"Video: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
              f"@ {self._fps:.1f} fps, {cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames")
        self._wall_start = time.time()
        return cap

    def _next(self, cap: cv2.VideoCapture) -> Optional[Tuple[np.ndarray, float]]:
        ok: bool
        frame: np.ndarray
        ok, frame = cap.read()
        if not ok:  # end of the video
            if not self.loop:
                print("Video ended")
                return None
            self._offset += self._index / self._fps  # keep timestamps increasing across loops
            self._index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                return None

        t: float = self._offset + self._index / self._fps  # the video's own time
        self._index += 1

        if self.realtime:  # wait until the real clock catches up with the video time
            delay: float = t - (time.time() - self._wall_start)
            if delay > 0:
                time.sleep(delay)
        return frame, t


def open_source(spec: str, loop: bool = False) -> VideoSource:
    """'0', '1', ... -> webcam with that index. Anything else -> a video file path."""
    if spec.isdigit():
        return CameraSource(cam_index=int(spec))
    return VideoFileSource(spec, realtime=True, loop=loop)