from typing import List, Optional, Tuple

import cv2
import numpy as np
import sys

import BodyDetector
import FaceDetector
from body_features import BodySignals, BodyWindowFeatures
from camera_source import CameraSource
from features import WindowFeatures
from video_source import VideoSource, open_source

# --- Display settings ---
VIEW_W: int = 640      # width of each video view in pixels
PANEL_H: int = 330     # height of each stats panel in pixels
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
WHITE: Tuple[int, int, int] = (255, 255, 255)
GREY: Tuple[int, int, int] = (190, 190, 190)
GREEN: Tuple[int, int, int] = (0, 200, 0)
RED: Tuple[int, int, int] = (0, 0, 255)
# --- Display settings ---
SOURCE: str = "videoplayback.mp4"      # "0" = first webcam, or a path such as "clips/syncope_01.mp4"
VIEW_W: int = 640      # width of each video view in pixels

Row = Tuple[str, str]  # (label, value text)


def percent(x: float) -> str:
    """0.37 -> '37%'"""
    return f"{x * 100:.0f}%"


def make_panel(title: str, status: str, status_ok: bool, rows: List[Row],
               width: int, height: int) -> np.ndarray:
    """Dark panel with a title, a colored status text, and one label/value row per line."""
    panel: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, title, (10, 28), FONT, 0.8, WHITE, 2)
    cv2.putText(panel, status, (width - 190, 28), FONT, 0.6, GREEN if status_ok else RED, 2)
    for i, (label, value) in enumerate(rows):
        y: int = 60 + i * 24
        cv2.putText(panel, label, (10, y), FONT, 0.55, GREY, 1)     # label on the left
        cv2.putText(panel, value, (400, y), FONT, 0.55, WHITE, 1)   # value in a fixed column
    return panel


def to_view(frame: Optional[np.ndarray], view_h: int) -> np.ndarray:
    """Resize a frame to the display size, or return a black image if there is no frame yet."""
    if frame is None:
        return np.zeros((view_h, VIEW_W, 3), dtype=np.uint8)
    return cv2.resize(frame, (VIEW_W, view_h))


def face_rows(ear: Optional[float], f: Optional[WindowFeatures]) -> List[Row]:
    """Text rows for the face panel. 'last 2 s' matches the extractor's default window."""
    rows: List[Row] = [("Eye openness now (EAR)", f"{ear:.2f}" if ear is not None else "--")]
    if f is None:
        rows.append(("Warming up...", ""))
        return rows
    rows += [
        ("Eyes closed (last 2 s)", percent(f.closed_fraction)),
        ("Lowest eye openness", f"{f.min_ear:.2f}"),
        ("Head pitch vs normal (deg)", f"{f.pitch_from_baseline:+.1f}"),
        ("Head pitch range (deg)", f"{f.pitch_range:.1f}"),
        ("Head motion speed", f"{f.motion_speed:.2f}"),
        ("Head shaking share (3-8 Hz)", percent(f.band_power_ratio)),
        ("Strongest head freq (Hz)", f"{f.dominant_freq_hz:.1f}"),
        ("Mouth asymmetry", f"{f.mean_abs_asymmetry:.3f}"),
        ("Face lost (last 2 s)", percent(f.face_lost_fraction)),
    ]
    return rows


def body_rows(f: Optional[BodyWindowFeatures]) -> List[Row]:
    """Text rows for the body panel. Distances are in shoulder widths."""
    if f is None:
        return [("Warming up...", "")]
    return [
        ("Head height vs shoulders", f"{f.head_offset_mean:+.2f}"),
        ("Head height range", f"{f.head_offset_range:.2f}"),
        ("Shoulder vertical range", f"{f.shoulder_y_range:.2f}"),
        ("Shoulder motion speed", f"{f.shoulder_motion_speed:.2f}"),
        ("Shoulder shaking share (3-8 Hz)", percent(f.band_power_ratio)),
        ("Strongest shoulder freq (Hz)", f"{f.dominant_freq_hz:.1f}"),
        ("Shoulder tilt (deg)", f"{f.shoulder_tilt_mean:+.1f}"),
        ("Wrist motion speed", f"{f.wrist_motion_speed:.2f}"),
        ("Hands visible", percent(f.hands_visible_fraction)),
        ("Wrist height difference", f"{f.wrist_height_diff:.2f}"),
        ("Body lost (last 2 s)", percent(f.pose_lost_fraction)),
    ]


if __name__ == "__main__":
    spec: str = sys.argv[1] if len(sys.argv) > 1 else SOURCE   # command line overrides the constant
    source: VideoSource = open_source(spec, loop=False)
    face: FaceDetector.FaceDetector = FaceDetector.FaceDetector(source)
    body: BodyDetector.BodyDetector = BodyDetector.BodyDetector(source)

    source.start()  # source first, then the two detectors that read from it
    face.start()
    body.start()
    try:
        # Main thread: display only (cv2.imshow must stay in the main thread)
        while True:
            face_frame: Optional[np.ndarray]
            ear: Optional[float]
            face_features: Optional[WindowFeatures]
            face_frame, ear, face_features = face.get_latest()

            body_frame: Optional[np.ndarray]
            body_signals: Optional[BodySignals]
            body_features: Optional[BodyWindowFeatures]
            body_frame, body_signals, body_features = body.get_latest()

            # Keep the aspect ratio of whichever frame is available (default 16:9)
            view_h: int = 360
            for fr in (face_frame, body_frame):
                if fr is not None:
                    view_h = int(fr.shape[0] * VIEW_W / fr.shape[1])
                    break

            face_found: bool = ear is not None
            body_found: bool = body_signals is not None and body_signals.pose_found

            face_panel: np.ndarray = make_panel("FACE", "Face detected" if face_found else "NO FACE",
                                                face_found, face_rows(ear, face_features), VIEW_W, PANEL_H)
            body_panel: np.ndarray = make_panel("BODY", "Body detected" if body_found else "NO BODY",
                                                body_found, body_rows(body_features), VIEW_W, PANEL_H)

            # Left column = face (video + stats), right column = body (video + stats).
            # cv2.resize makes a new image, so we never draw on the detectors' own frames.
            top: np.ndarray = np.hstack([to_view(face_frame, view_h), to_view(body_frame, view_h)])
            bottom: np.ndarray = np.hstack([face_panel, body_panel])
            cv2.imshow("CabinGuard monitor", np.vstack([top, bottom]))

            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
                break
    finally:
        face.stop()
        body.stop()
        source.stop()  # stop the camera last, after its readers
        cv2.destroyAllWindows()