import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, List, Optional, Tuple

import numpy as np

Landmarks = List[Any]          # MediaPipe pose landmarks (33 items, each has x, y, z, visibility)
Point = Tuple[float, float]    # (x, y)

# --- Pose landmark indices (33-point body model) ---
NOSE: int = 0
L_SHOULDER: int = 11           # the person's left shoulder
R_SHOULDER: int = 12
L_WRIST: int = 15
R_WRIST: int = 16

MIN_VISIBILITY: float = 0.3    # below this, a landmark is treated as "not seen"


# ======================= per-frame signals =======================

@dataclass
class BodySignals:
    """Raw measurements from ONE frame. None means 'not available'.
    All positions are in SHOULDER-WIDTH units, so camera distance matters less."""
    t: float                          # time in seconds
    pose_found: bool                  # both shoulders AND the nose were visible
    shoulder_tilt: Optional[float]    # shoulder line angle in degrees (sign: verify by testing)
    shoulder_x: Optional[float]       # midpoint between the shoulders
    shoulder_y: Optional[float]
    head_offset: Optional[float]      # (nose_y - shoulder_mid_y): negative = head above shoulders
    left_wrist: Optional[Point]       # wrist position relative to shoulder midpoint, None if not visible
    right_wrist: Optional[Point]


def _px(points: Landmarks, i: int, w: int, h: int) -> np.ndarray:
    """Landmark i as a pixel position [x, y]."""
    return np.array([points[i].x * w, points[i].y * h])


def _visible(points: Landmarks, i: int) -> bool:
    """True if the model is confident that landmark i is actually in view."""
    v: Optional[float] = points[i].visibility
    return v is not None and v >= MIN_VISIBILITY


def build_body_signals(t: float, result: Any, w: int, h: int) -> BodySignals:
    """Turn one MediaPipe pose result into a BodySignals."""
    not_found: BodySignals = BodySignals(t, False, None, None, None, None, None, None)

    if not result.pose_landmarks:
        return not_found
    points: Landmarks = result.pose_landmarks[0]  # first (only) person

    # We need both shoulders and the nose to define the basic posture
    if not (_visible(points, L_SHOULDER) and _visible(points, R_SHOULDER) and _visible(points, NOSE)):
        return not_found

    l_sh: np.ndarray = _px(points, L_SHOULDER, w, h)
    r_sh: np.ndarray = _px(points, R_SHOULDER, w, h)
    sw: float = float(np.linalg.norm(l_sh - r_sh))  # shoulder width in pixels (our unit of length)
    if sw < 1e-6:
        return not_found

    mid: np.ndarray = (l_sh + r_sh) / 2.0
    d: np.ndarray = l_sh - r_sh
    # abs() on x avoids a +-180 degree jump. Positive when the person's left shoulder is lower
    # (image y points down) - verify the sign by tilting.
    tilt: float = math.degrees(math.atan2(d[1], abs(d[0])))

    nose: np.ndarray = _px(points, NOSE, w, h)
    head_offset: float = float((nose[1] - mid[1]) / sw)

    def wrist(i: int) -> Optional[Point]:
        """Wrist position relative to the shoulder midpoint, in shoulder widths."""
        if not _visible(points, i):
            return None
        rel: np.ndarray = (_px(points, i, w, h) - mid) / sw
        return (float(rel[0]), float(rel[1]))

    return BodySignals(t=t, pose_found=True, shoulder_tilt=tilt,
                       shoulder_x=float(mid[0] / sw), shoulder_y=float(mid[1] / sw),
                       head_offset=head_offset,
                       left_wrist=wrist(L_WRIST), right_wrist=wrist(R_WRIST))


# ======================= helpers =======================

def band_power(signals: List[np.ndarray], t: np.ndarray,
               band_hz: Tuple[float, float]) -> Tuple[float, float]:
    """(share of power inside band_hz, dominant frequency in Hz) for the given signals.
    Same idea as in features.py: resample evenly, FFT, ignore drift below 0.5 Hz."""
    duration: float = float(t[-1] - t[0])
    if len(t) < 16 or duration <= 0:
        return 0.0, 0.0
    fs: float = (len(t) - 1) / duration
    grid: np.ndarray = np.arange(t[0], t[-1], 1.0 / fs)
    total: Optional[np.ndarray] = None
    for sig in signals:
        u: np.ndarray = np.interp(grid, t, sig)
        u = (u - u.mean()) * np.hanning(len(u))
        p: np.ndarray = np.abs(np.fft.rfft(u)) ** 2
        total = p if total is None else total + p
    freqs: np.ndarray = np.fft.rfftfreq(len(grid), d=1.0 / fs)
    above_drift: np.ndarray = freqs > 0.5
    hi: float = min(band_hz[1], 0.5 * fs)
    in_band: np.ndarray = (freqs >= band_hz[0]) & (freqs <= hi)
    denom: float = float(total[above_drift].sum())
    if denom < 1e-12:
        return 0.0, 0.0
    return (float(total[in_band].sum() / denom),
            float(freqs[above_drift][np.argmax(total[above_drift])]))


def mean_speed(samples: List[Tuple[float, Point]]) -> Optional[float]:
    """Mean speed of a point given (time, (x, y)) samples. None if fewer than 2 samples."""
    if len(samples) < 2:
        return None
    t: np.ndarray = np.array([s[0] for s in samples])
    xy: np.ndarray = np.array([s[1] for s in samples])
    dt: np.ndarray = np.maximum(np.diff(t), 1e-3)
    return float(np.mean(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) / dt))


# ======================= window features =======================

@dataclass
class BodyWindowFeatures:
    """Features for one sliding window of body data."""
    pose_lost_fraction: float         # fraction of ALL frames without a usable pose
    longest_lost_s: float             # longest stretch without a pose
    time_since_pose_s: float          # time since the pose was last seen
    head_offset_mean: float           # near 0 or positive = head dropped onto the shoulders (slump cue)
    head_offset_range: float          # max - min of head offset
    shoulder_y_range: float           # vertical movement of the shoulders
    shoulder_motion_speed: float      # mean shoulder speed, shoulder-widths per second
    band_power_ratio: float           # share of shoulder motion power in the 3-8 Hz band (seizure cue)
    dominant_freq_hz: float           # strongest shoulder motion frequency
    shoulder_tilt_mean: float         # sideways slump cue
    shoulder_tilt_range: float
    wrist_motion_speed: float         # mean wrist speed relative to the body
    hands_visible_fraction: float     # fraction of frames with at least one wrist visible
    wrist_height_diff: float          # mean |left - right wrist height| (rough arm-drift proxy)

    def to_vector(self) -> List[float]:
        """Fixed-order list of numbers for scikit-learn."""
        return [float(v) for v in asdict(self).values()]


class BodyWindowFeatureExtractor:
    """Keeps the last `window_s` seconds of BodySignals and computes features."""

    def __init__(self, window_s: float = 2.0, band_hz: Tuple[float, float] = (3.0, 8.0),
                 min_frames: int = 10) -> None:
        self.window_s: float = window_s
        self.band_hz: Tuple[float, float] = band_hz
        self.min_frames: int = min_frames
        self._buf: Deque[BodySignals] = deque()

    def add(self, s: BodySignals) -> None:
        """Add a frame and drop frames older than the window."""
        self._buf.append(s)
        while self._buf and s.t - self._buf[0].t > self.window_s:
            self._buf.popleft()

    def compute(self) -> Optional[BodyWindowFeatures]:
        """Features for the current window, or None if too few frames yet."""
        frames: List[BodySignals] = list(self._buf)
        n: int = len(frames)
        if n < self.min_frames:
            return None

        found: List[BodySignals] = [f for f in frames if f.pose_found]
        lost_fraction: float = 1.0 - len(found) / n

        # --- longest stretch without a pose ---
        longest_lost: float = 0.0
        run_start: Optional[float] = None
        for f in frames:
            if not f.pose_found:
                if run_start is None:
                    run_start = f.t
                longest_lost = max(longest_lost, f.t - run_start)
            else:
                run_start = None
        time_since: float = frames[-1].t - found[-1].t if found else self.window_s

        if len(found) < self.min_frames:
            # Not enough body data: report the loss, zeros everywhere else
            return BodyWindowFeatures(
                pose_lost_fraction=lost_fraction, longest_lost_s=longest_lost,
                time_since_pose_s=time_since, head_offset_mean=0.0, head_offset_range=0.0,
                shoulder_y_range=0.0, shoulder_motion_speed=0.0, band_power_ratio=0.0,
                dominant_freq_hz=0.0, shoulder_tilt_mean=0.0, shoulder_tilt_range=0.0,
                wrist_motion_speed=0.0, hands_visible_fraction=0.0, wrist_height_diff=0.0)

        t: np.ndarray = np.array([f.t for f in found])
        head: np.ndarray = np.array([f.head_offset for f in found], dtype=float)
        sx: np.ndarray = np.array([f.shoulder_x for f in found], dtype=float)
        sy: np.ndarray = np.array([f.shoulder_y for f in found], dtype=float)
        tilt: np.ndarray = np.array([f.shoulder_tilt for f in found], dtype=float)

        # --- shoulders: speed and frequency of the midpoint ---
        dt: np.ndarray = np.maximum(np.diff(t), 1e-3)
        shoulder_speed: float = float(np.mean(np.hypot(np.diff(sx), np.diff(sy)) / dt))
        band_ratio: float
        dom_freq: float
        band_ratio, dom_freq = band_power([sx, sy], t, self.band_hz)

        # --- wrists ---
        left_samples: List[Tuple[float, Point]] = [(f.t, f.left_wrist) for f in found if f.left_wrist is not None]
        right_samples: List[Tuple[float, Point]] = [(f.t, f.right_wrist) for f in found if f.right_wrist is not None]
        speeds: List[float] = [s for s in (mean_speed(left_samples), mean_speed(right_samples)) if s is not None]
        wrist_speed: float = float(np.mean(speeds)) if speeds else 0.0

        hands_visible: float = float(np.mean(
            [f.left_wrist is not None or f.right_wrist is not None for f in found]))
        both: List[float] = [abs(f.left_wrist[1] - f.right_wrist[1]) for f in found
                             if f.left_wrist is not None and f.right_wrist is not None]
        wrist_diff: float = float(np.mean(both)) if both else 0.0

        return BodyWindowFeatures(
            pose_lost_fraction=lost_fraction, longest_lost_s=longest_lost, time_since_pose_s=time_since,
            head_offset_mean=float(head.mean()), head_offset_range=float(head.max() - head.min()),
            shoulder_y_range=float(sy.max() - sy.min()), shoulder_motion_speed=shoulder_speed,
            band_power_ratio=band_ratio, dominant_freq_hz=dom_freq,
            shoulder_tilt_mean=float(tilt.mean()), shoulder_tilt_range=float(tilt.max() - tilt.min()),
            wrist_motion_speed=wrist_speed, hands_visible_fraction=hands_visible,
            wrist_height_diff=wrist_diff)