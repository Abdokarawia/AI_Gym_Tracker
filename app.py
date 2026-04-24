"""
AI Gym Tracker — Streamlit Real-Time App  (REDESIGNED)
Uses streamlit-webrtc for true browser-side webcam (works on Streamlit Cloud)
Supports: Squat · Push-Up · Pull-Up · Jumping Jack · Russian Twist

Features:
  • Real-time pose estimation via MediaPipe Heavy model
  • Per-rep form scoring & error history panel
  • Exercise mismatch detection / warning
  • Professional dark-sport UI (Inter + monospace numerals)
  • Angle history chart after video-upload analysis
"""

# ── streamlit-webrtc shutdown-observer defensive patch ───────────────────────
# Guards against AttributeError: '_polling_thread' when stop() is called
# before __init__ completes (bug in older streamlit-webrtc on Python 3.12+).
# Safe no-op if the library version already has the upstream fix (>=0.48).
try:
    from streamlit_webrtc.shutdown import SessionShutdownObserver as _SSO
    _sso_orig_stop = _SSO.stop

    def _sso_safe_stop(self):
        if not hasattr(self, "_polling_thread") or self._polling_thread is None:
            return
        try:
            _sso_orig_stop(self)
        except AttributeError:
            pass
        except Exception:
            pass

    _SSO.stop = _sso_safe_stop
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import cv2
import numpy as np
import math
import os
import tempfile
import threading
from collections import deque
from av import VideoFrame
from aiortc import RTCConfiguration as AioRTCConfiguration, RTCIceServer
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Gym Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark sport aesthetic ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg0:   #0a0c10;
    --bg1:   #10141c;
    --bg2:   #171d28;
    --bg3:   #1e2635;
    --line:  rgba(255,255,255,0.07);
    --acc:   #00e5a0;
    --acc2:  #ff5c3a;
    --acc3:  #ffce3a;
    --txt:   #e8ecf2;
    --muted: #6b7a92;
    --font:  'Barlow', sans-serif;
    --cond:  'Barlow Condensed', sans-serif;
    --mono:  'JetBrains Mono', monospace;
}

html, body, .stApp { background: var(--bg0) !important; color: var(--txt) !important; font-family: var(--font); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; font-family: var(--font) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--cond) !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr { border-color: var(--line) !important; }

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--bg2) !important;
    border-color: var(--line) !important;
    color: var(--txt) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: var(--txt) !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { color: var(--txt) !important; }

/* primary button (Reset) */
[data-testid="stSidebar"] .stButton > button {
    background: var(--bg2) !important;
    color: var(--txt) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    font-family: var(--cond) !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 13px !important;
    transition: background .15s, border-color .15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg3) !important;
    border-color: var(--acc) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--acc) !important;
    color: #000 !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    opacity: .88;
}

/* ── Main area ── */
.block-container { padding-top: 1.4rem !important; padding-bottom: 2rem !important; }

/* Page title */
h1 { font-family: var(--cond) !important; font-size: 2.4rem !important; font-weight: 900 !important;
     letter-spacing: 3px; text-transform: uppercase; color: var(--acc) !important; margin-bottom: 0 !important; }
.subtitle { font-family: var(--font); font-size: 13px; color: var(--muted); letter-spacing: .5px; margin-top: -6px; }

/* ── Warning box ── */
.warn-mismatch {
    background: rgba(255,92,58,0.10);
    border: 1px solid rgba(255,92,58,0.45);
    border-left: 4px solid var(--acc2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0 16px;
    font-size: 13.5px;
    color: #ffc4b8;
    line-height: 1.55;
}
.warn-mismatch strong { color: var(--acc2); }
.warn-ok {
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.3);
    border-left: 4px solid var(--acc);
    border-radius: 8px;
    padding: 10px 16px;
    margin: 10px 0 16px;
    font-size: 13px;
    color: #9de8cc;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 10px; margin-bottom: 12px; }
.metric-card {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 10px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--acc);
}
.metric-card.orange::before { background: var(--acc2); }
.metric-card.blue::before   { background: #4e8ef7; }
.metric-card.yellow::before { background: var(--acc3); }
.metric-value {
    font-family: var(--mono);
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--acc);
    line-height: 1.1;
    letter-spacing: -1px;
}
.metric-card.orange .metric-value { color: var(--acc2); }
.metric-card.blue   .metric-value { color: #4e8ef7; }
.metric-card.yellow .metric-value { color: var(--acc3); }
.metric-label {
    font-family: var(--cond);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Feedback box ── */
.feedback-card {
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    color: var(--txt);
    line-height: 1.5;
    margin-bottom: 12px;
}
.feedback-card .fb-label {
    font-family: var(--cond);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.feedback-card .fb-msg { color: var(--acc); font-weight: 600; }

/* ── History panel ── */
.hist-header {
    font-family: var(--cond);
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
    margin-bottom: 10px;
}
.hist-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 13px;
}
.hist-rep-num {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    min-width: 28px;
}
.hist-bar-bg {
    flex: 1;
    background: var(--bg3);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.hist-bar-fill { height: 100%; border-radius: 4px; }
.hist-score {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    min-width: 34px;
    text-align: right;
}
.hist-tag {
    font-family: var(--cond);
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 20px;
    white-space: nowrap;
}
.tag-good   { background: rgba(0,229,160,0.12);  color: var(--acc);  border: 1px solid rgba(0,229,160,0.3);  }
.tag-ok     { background: rgba(255,206,58,0.12); color: var(--acc3); border: 1px solid rgba(255,206,58,0.3); }
.tag-poor   { background: rgba(255,92,58,0.12);  color: var(--acc2); border: 1px solid rgba(255,92,58,0.3);  }
.hist-error-note {
    font-size: 11px;
    color: var(--acc2);
    margin-top: 2px;
    display: block;
}

/* ── Session summary ── */
.session-summary {
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.sess-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.sess-label { font-family: var(--cond); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); }
.sess-val { font-family: var(--mono); font-size: 16px; font-weight: 700; }

/* ── End button ── */
.end-btn-wrap a {
    display: block; width: 100%;
    padding: 11px 0;
    background: var(--acc);
    color: #000 !important;
    text-align: center;
    border-radius: 8px;
    font-family: var(--cond);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-decoration: none;
    margin-top: 8px;
    transition: opacity .15s;
}
.end-btn-wrap a:hover { opacity: .85; }

/* ── Exercise tag chips in sidebar ── */
.ex-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.ex-chip {
    font-family: var(--cond);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: var(--bg2);
    border: 1px solid var(--line);
    color: var(--muted);
    padding: 3px 10px;
    border-radius: 20px;
}

/* ── Streamlit widget overrides ── */
.stRadio > label { color: var(--muted) !important; font-size: 13px !important; }
.stCheckbox > label { color: var(--txt) !important; }
.stSelectbox > label { color: var(--muted) !important; font-size: 12px !important;
                        font-family: var(--cond) !important; letter-spacing: 1.5px; text-transform: uppercase; }
div[data-testid="stCaption"] { color: var(--muted) !important; }

/* ── Line chart override ── */
[data-testid="stVegaLiteChart"] { background: var(--bg2) !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── MediaPipe model ───────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_heavy.task"

@st.cache_resource(show_spinner="Downloading MediaPipe model (~25 MB)…")
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
        urllib.request.urlretrieve(url, MODEL_PATH)
    return MODEL_PATH

ensure_model()


# ── ICE / TURN configuration ──────────────────────────────────────────────────
def _secret(key: str, fallback: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, fallback)

_TURN_HOST = _secret("TURN_SERVER",     "openrelay.metered.ca")
_TURN_USER = _secret("TURN_USERNAME",   "openrelayproject")
_TURN_PASS = _secret("TURN_CREDENTIAL", "openrelayproject")

SERVER_RTC_CONFIG = AioRTCConfiguration(iceServers=[
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(
        urls=[f"turn:{_TURN_HOST}:443?transport=tcp"],
        username=_TURN_USER,
        credential=_TURN_PASS,
    ),
])

FRONTEND_RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
        ]},
        {"urls": [
            f"turn:{_TURN_HOST}:80",
            f"turn:{_TURN_HOST}:443",
            f"turn:{_TURN_HOST}:443?transport=tcp",
        ],
         "username":   _TURN_USER,
         "credential": _TURN_PASS,
        },
    ]
})


# ── Pose utilities ────────────────────────────────────────────────────────────
LM = {
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow':    13, 'r_elbow':    14,
    'l_wrist':    15, 'r_wrist':    16,
    'l_hip':      23, 'r_hip':      24,
    'l_knee':     25, 'r_knee':     26,
    'l_ankle':    27, 'r_ankle':    28,
}
SKELETON_EDGES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
]

def lm_px(lms, idx, w, h):
    lm = lms[idx]
    return int(lm.x * w), int(lm.y * h)

def angle3(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))

def _angle_score(best, ideal, worst):
    if ideal < worst:
        return max(0, min(100, int((worst - best) / (worst - ideal) * 100)))
    return max(0, min(100, int((best - worst) / (ideal - worst) * 100)))


# ── Rep counter ───────────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self, up_thresh, down_thresh, higher_is_up=True):
        self.up_thresh    = up_thresh
        self.down_thresh  = down_thresh
        self.higher_is_up = higher_is_up
        self.count        = 0
        self.stage        = None
        self.history      = deque(maxlen=2000)
        self._rep_best    = None
        self._angle_scores = []

    def update(self, v):
        self.history.append(v)
        if self._rep_best is None:
            self._rep_best = v
        if self.higher_is_up:
            self._rep_best = min(self._rep_best, v)
            if v > self.up_thresh:
                self.stage = 'up'; self._rep_best = None
            elif v < self.down_thresh and self.stage == 'up':
                self.stage = 'down'; self.count += 1; return 'rep'
        else:
            self._rep_best = max(self._rep_best, v)
            if v < self.up_thresh:
                self.stage = 'up'; self._rep_best = None
            elif v > self.down_thresh and self.stage == 'up':
                self.stage = 'down'; self.count += 1; return 'rep'
        return ''

    def angle_score(self):
        return int(sum(self._angle_scores) / len(self._angle_scores)) if self._angle_scores else 0

    def reset(self):
        self.count = 0; self.stage = None
        self.history.clear(); self._rep_best = None; self._angle_scores = []


# ── Exercise analyzers ────────────────────────────────────────────────────────
class SquatAnalyzer:
    label       = 'Squat'
    pose_hints  = ['deep knee bend', 'thighs parallel to ground']

    def __init__(self): self.rc = RepCounter(160, 90)

    def analyze(self, lms, w, h):
        hip   = lm_px(lms, LM['l_hip'],   w, h)
        knee  = lm_px(lms, LM['l_knee'],  w, h)
        ankle = lm_px(lms, LM['l_ankle'], w, h)
        ang = angle3(hip, knee, ankle)
        rep = self.rc.update(ang)
        error = None
        if rep == 'rep' and self.rc._rep_best is not None:
            sc = _angle_score(self.rc._rep_best, 70, 160)
            self.rc._angle_scores.append(sc)
            if sc < 50:
                error = "Depth too shallow — go lower next rep"
        fb = ('Go lower — knees past 90°!'  if ang > 110 and self.rc.stage != 'down'
              else 'Perfect depth! 🔥'       if ang < 90
              else 'Drive through heels!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score(),
                    last_error=error, rep_triggered=(rep == 'rep'))


class PushUpAnalyzer:
    label       = 'Push-Up'
    pose_hints  = ['plank position', 'arms bending toward floor']

    def __init__(self): self.rc = RepCounter(155, 90)

    def analyze(self, lms, w, h):
        sh = lm_px(lms, LM['l_shoulder'], w, h)
        el = lm_px(lms, LM['l_elbow'],   w, h)
        wr = lm_px(lms, LM['l_wrist'],   w, h)
        ang = angle3(sh, el, wr)
        rep = self.rc.update(ang)
        error = None
        if rep == 'rep' and self.rc._rep_best is not None:
            sc = _angle_score(self.rc._rep_best, 60, 155)
            self.rc._angle_scores.append(sc)
            if sc < 50:
                error = "Elbow angle too high — lower chest to ground"
        fb = ('Lower chest to floor!'    if ang > 130 and self.rc.stage != 'down'
              else 'Full range! 🔥'      if ang < 90
              else 'Push up strong!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score(),
                    last_error=error, rep_triggered=(rep == 'rep'))


class PullUpAnalyzer:
    label       = 'Pull-Up'
    pose_hints  = ['arms fully extended overhead', 'hanging from bar']

    def __init__(self): self.rc = RepCounter(50, 140, higher_is_up=False)

    def analyze(self, lms, w, h):
        sh = lm_px(lms, LM['l_shoulder'], w, h)
        el = lm_px(lms, LM['l_elbow'],   w, h)
        wr = lm_px(lms, LM['l_wrist'],   w, h)
        ang = angle3(sh, el, wr)
        rep = self.rc.update(ang)
        error = None
        if rep == 'rep' and self.rc._rep_best is not None:
            sc = _angle_score(self.rc._rep_best, 30, 140)
            self.rc._angle_scores.append(sc)
            if sc < 50:
                error = "Chin did not clear bar — pull higher"
        fb = ('Pull higher — chin over bar!'    if ang > 70 and self.rc.stage == 'up'
              else 'Chin over bar! 🔥'          if ang < 50
              else 'Lower with control!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score(),
                    last_error=error, rep_triggered=(rep == 'rep'))


class JumpingJackAnalyzer:
    label       = 'Jumping Jack'
    pose_hints  = ['arms raised wide', 'feet jumping apart']

    def __init__(self): self.rc = RepCounter(130, 40); self._rep_max = 0

    def analyze(self, lms, w, h):
        lsh = lm_px(lms, LM['l_shoulder'], w, h)
        lhi = lm_px(lms, LM['l_hip'],     w, h)
        lwr = lm_px(lms, LM['l_wrist'],   w, h)
        ang = angle3(lhi, lsh, lwr)
        self._rep_max = max(self._rep_max, ang)
        rep = self.rc.update(ang)
        error = None
        if rep == 'rep':
            sc = _angle_score(self._rep_max, 150, 40)
            self.rc._angle_scores.append(sc)
            if sc < 50:
                error = "Arms not reaching overhead — extend fully"
            self._rep_max = 0
        fb = 'Arms overhead!' if ang < 80 else ('Full extension! 🔥' if ang > 120 else 'Keep moving!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score(),
                    last_error=error, rep_triggered=(rep == 'rep'))


class RussianTwistAnalyzer:
    label       = 'Russian Twist'
    pose_hints  = ['seated lean-back', 'hands moving side to side']

    def __init__(self):
        self.rc = RepCounter(30, 5)
        self._last_side = None; self._touches = 0; self._touch_max = 0

    def analyze(self, lms, w, h):
        lsh = lm_px(lms, LM['l_shoulder'], w, h)
        rsh = lm_px(lms, LM['r_shoulder'], w, h)
        lhi = lm_px(lms, LM['l_hip'],     w, h)
        rhi = lm_px(lms, LM['r_hip'],     w, h)
        lwr = lm_px(lms, LM['l_wrist'],   w, h)
        rwr = lm_px(lms, LM['r_wrist'],   w, h)
        scx  = (lsh[0] + rsh[0]) / 2
        hcx  = (lhi[0] + rhi[0]) / 2
        rot  = abs(scx - hcx)
        wcx  = (lwr[0] + rwr[0]) / 2
        side = 'left' if wcx < hcx else 'right'
        self._touch_max = max(self._touch_max, rot)
        error = None
        if rot > 30 and side != self._last_side:
            self._last_side = side; self._touches += 1
            if self._touches % 2 == 0:
                self.rc.count += 1
                sc = _angle_score(self._touch_max, 80, 0)
                self.rc._angle_scores.append(sc)
                if sc < 50:
                    error = "Rotation range too small — twist further"
                self._touch_max = 0
        self.rc.history.append(rot)
        fb = f'Twist {side}!' if rot < 20 else f'Good rotation → {side}! 🔥'
        return dict(angle=rot, stage=side, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score(),
                    last_error=error, rep_triggered=False)


ANALYZERS = {
    'Squat':         SquatAnalyzer,
    'Push-Up':       PushUpAnalyzer,
    'Pull-Up':       PullUpAnalyzer,
    'Jumping Jack':  JumpingJackAnalyzer,
    'Russian Twist': RussianTwistAnalyzer,
}

# ── Pose-hint heuristics for exercise mismatch detection ─────────────────────
# Simple joint-angle heuristics to guess what the person is actually doing.
def _guess_exercise(lms, w, h):
    """Return a best-guess exercise name from landmarks, or None if uncertain."""
    try:
        hip   = lm_px(lms, LM['l_hip'],   w, h)
        knee  = lm_px(lms, LM['l_knee'],  w, h)
        ankle = lm_px(lms, LM['l_ankle'], w, h)
        sh    = lm_px(lms, LM['l_shoulder'], w, h)
        el    = lm_px(lms, LM['l_elbow'],   w, h)
        wr    = lm_px(lms, LM['l_wrist'],   w, h)
        lhi   = lm_px(lms, LM['l_hip'],     w, h)
        rhi   = lm_px(lms, LM['r_hip'],     w, h)

        knee_ang  = angle3(hip, knee, ankle)
        elbow_ang = angle3(sh, el, wr)

        # Wrist height relative to shoulder for overhead arm detection
        wrist_above_sh = wr[1] < sh[1]

        # Hip height — low hip y means person is lower to ground
        hip_low = hip[1] > h * 0.6

        # Trunk tilt — shoulder y vs hip y (plank = close vertical distance)
        sh_y, hi_y = sh[1], ((lhi[1] + rhi[1]) / 2)
        roughly_horizontal = abs(sh_y - hi_y) < h * 0.15

        if roughly_horizontal and elbow_ang < 130:
            return 'Push-Up'
        if wrist_above_sh and not roughly_horizontal:
            if elbow_ang < 100:
                return 'Pull-Up'
            return 'Jumping Jack'
        if knee_ang < 130 and hip_low:
            return 'Squat'
        return None
    except Exception:
        return None


# ── Drawing helpers ───────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_DUPLEX
ACC_GREEN  = (0, 229, 160)
ACC_ORANGE = (58, 92, 255)   # BGR
ACC_RED    = (58, 92, 255)
WHITE      = (232, 236, 242)
DARK       = (16, 20, 28)

def draw_skeleton(frame, lms, w, h):
    for a, b in SKELETON_EDGES:
        cv2.line(frame, lm_px(lms, a, w, h), lm_px(lms, b, w, h),
                 (0, 200, 130), 2, cv2.LINE_AA)
    for i in range(33):
        pt = lm_px(lms, i, w, h)
        cv2.circle(frame, pt, 5, (0, 229, 160), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255),  1, cv2.LINE_AA)


def _pill(frame, text, pos, font_scale=0.5, thickness=1,
          bg=(30, 38, 55), fg=(232, 236, 242), pad=(10, 6)):
    """Draw a rounded pill label."""
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    x1, y1 = x - pad[0], y - th - pad[1]
    x2, y2 = x + tw + pad[0], y + pad[1]
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, font_scale, fg, thickness, cv2.LINE_AA)


def draw_hud(frame, res, ex_title, mismatch_ex=None):
    H, W = frame.shape[:2]

    count    = res.get('count', 0)
    stage    = res.get('stage', '') or ''
    feedback = res.get('feedback', '')
    angle_v  = float(res.get('angle') or 0)
    score    = res.get('form_score', 0)

    # Semi-transparent overlay strip top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 54), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Exercise label (top center)
    label = ex_title.upper()
    lw, _ = cv2.getTextSize(label, FONT, 0.65, 1)[0]
    cv2.putText(frame, label, (W//2 - lw//2, 34),
                FONT, 0.65, (0, 229, 160), 1, cv2.LINE_AA)

    # Mismatch warning strip
    if mismatch_ex:
        warn_txt = f"! Detected: {mismatch_ex.upper()} — check selection"
        ww, _ = cv2.getTextSize(warn_txt, FONT, 0.42, 1)[0]
        cv2.rectangle(frame, (0, 54), (W, 78), (20, 10, 10), -1)
        cv2.putText(frame, warn_txt, (W//2 - ww//2, 70),
                    FONT, 0.42, (255, 100, 80), 1, cv2.LINE_AA)

    # Left panel — REPS
    cv2.rectangle(frame, (8, 60), (130, 185), (16, 22, 34), -1)
    cv2.rectangle(frame, (8, 60), (130, 64),  (0, 229, 160), -1)
    cv2.putText(frame, 'REPS', (20, 82), FONT, 0.38, (107, 122, 146), 1, cv2.LINE_AA)
    cstr = str(count)
    cw, _ = cv2.getTextSize(cstr, FONT, 2.8, 2)[0]
    cv2.putText(frame, cstr, (69 - cw//2, 168), FONT, 2.8,
                (10, 14, 22), 6, cv2.LINE_AA)
    cv2.putText(frame, cstr, (69 - cw//2, 168), FONT, 2.8,
                (0, 229, 160), 2, cv2.LINE_AA)

    # Stage chip
    sl = stage.upper() if stage else 'READY'
    sc = (0, 180, 110) if stage == 'up' else (58, 100, 220)
    cv2.rectangle(frame, (8, 190), (130, 215), sc, -1)
    sw, _ = cv2.getTextSize(sl, FONT, 0.42, 1)[0]
    cv2.putText(frame, sl, (69 - sw//2, 207), FONT, 0.42, (10, 14, 22), 2, cv2.LINE_AA)

    # Right panel — ANGLE
    cv2.rectangle(frame, (W - 138, 60), (W - 8, 185), (16, 22, 34), -1)
    cv2.rectangle(frame, (W - 138, 60), (W - 8, 64), (255, 92, 58), -1)
    cv2.putText(frame, 'ANGLE', (W - 128, 82), FONT, 0.38, (107, 122, 146), 1, cv2.LINE_AA)
    astr = f'{int(angle_v)}'
    aw, _ = cv2.getTextSize(astr, FONT, 1.8, 2)[0]
    cv2.putText(frame, astr, (W - 73 - aw//2, 152), FONT, 1.8,
                (10, 14, 22), 5, cv2.LINE_AA)
    cv2.putText(frame, astr, (W - 73 - aw//2, 152), FONT, 1.8,
                (255, 165, 58), 2, cv2.LINE_AA)
    cv2.putText(frame, 'deg', (W - 73 - aw//2 + aw + 4, 152),
                FONT, 0.38, (107, 122, 146), 1, cv2.LINE_AA)

    # Form score bar (bottom-left)
    bar_x, bar_y, bar_w = 8, H - 28, 200
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (22, 30, 44), -1)
    fill = int(bar_w * score / 100) if score else 0
    bar_col = (0, 229, 160) if score >= 75 else (58, 200, 255) if score >= 50 else (58, 92, 255)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 10), bar_col, -1)
    cv2.putText(frame, f'FORM {score}', (bar_x, bar_y - 4),
                FONT, 0.35, (107, 122, 146), 1, cv2.LINE_AA)

    # Feedback (bottom center)
    if feedback:
        fb = feedback[:60]
        fw, fh = cv2.getTextSize(fb, FONT, 0.48, 1)[0]
        fy = H - 12
        overlay2 = frame.copy()
        cv2.rectangle(overlay2,
                      (W//2 - fw//2 - 16, fy - fh - 14),
                      (W//2 + fw//2 + 16, fy + 4),
                      (10, 14, 22), -1)
        cv2.addWeighted(overlay2, 0.88, frame, 0.12, 0, frame)
        cv2.putText(frame, fb, (W//2 - fw//2, fy),
                    FONT, 0.48, (0, 229, 160), 1, cv2.LINE_AA)


# ── Shared gym state ──────────────────────────────────────────────────────────
class GymState:
    def __init__(self):
        self.lock             = threading.Lock()
        self.result           = {'count': 0, 'stage': '', 'feedback': 'Get in position!',
                                 'angle': 0, 'form_score': 0,
                                 'last_error': None, 'rep_triggered': False}
        self.exercise         = 'Squat'
        self.show_skeleton    = True
        self.mirror           = True
        self.analyzer         = SquatAnalyzer()
        self.detected_exercise  = None      # live pose guess
        self.rep_history        = []         # list of dicts {rep, score, error}
        self._mp                = None
        self._landmarker        = None
        self._mismatch_active   = False      # True while wrong exercise is ongoing
        self._mismatch_frames   = 0          # consecutive frames of mismatch
        self._MISMATCH_TRIGGER  = 15         # frames before logging one event (~0.5s)

    def set_exercise(self, ex):
        with self.lock:
            if ex != self.exercise:
                self.exercise  = ex
                self.analyzer  = ANALYZERS[ex]()
                self.result    = {'count': 0, 'stage': '', 'feedback': 'Get in position!',
                                  'angle': 0, 'form_score': 0,
                                  'last_error': None, 'rep_triggered': False}
                self.rep_history      = []
                self._mismatch_active = False
                self._mismatch_frames = 0
                self._close_landmarker()

    def _close_landmarker(self):
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def get_landmarker(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            self._mp = mp
            opts = PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._landmarker = PoseLandmarker.create_from_options(opts)
        return self._mp, self._landmarker

    def process_frame(self, frame_bgr):
        with self.lock:
            mp, landmarker = self.get_landmarker()
            H, W = frame_bgr.shape[:2]
            rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det    = landmarker.detect(mp_img)

            mismatch_ex = None
            if det.pose_landmarks:
                lms = det.pose_landmarks[0]
                # Heuristic exercise guess
                guess = _guess_exercise(lms, W, H)
                self.detected_exercise = guess
                if guess and guess != self.exercise:
                    mismatch_ex = guess

                if self.show_skeleton:
                    draw_skeleton(frame_bgr, lms, W, H)
                self.result = self.analyzer.analyze(lms, W, H)

                # ── Mismatch penalty ─────────────────────────────────────────
                # Only log ONE history entry per continuous wrong-exercise bout.
                # A bout starts when mismatch is first detected and ends when
                # the correct exercise (or no pose) is seen again.
                if mismatch_ex:
                    self._mismatch_frames += 1
                    # Undo any rep that fired during the wrong movement
                    if self.result.get('rep_triggered'):
                        self.analyzer.rc.count = max(0, self.analyzer.rc.count - 1)
                        if self.analyzer.rc._angle_scores:
                            self.analyzer.rc._angle_scores.pop()
                    # Penalty score: 30–45 (poor, not zero) to reflect partial effort
                    import random
                    penalty_score = random.randint(30, 45)
                    penalty_msg = (
                        f"Wrong exercise! Doing {mismatch_ex} "
                        f"but '{self.exercise}' is selected"
                    )
                    self.result = {
                        **self.result,
                        'count':         self.analyzer.rc.count,
                        'form_score':    penalty_score,
                        'feedback':      penalty_msg,
                        'last_error':    penalty_msg,
                        'rep_triggered': False,
                    }
                    # Log exactly ONE entry when the bout first triggers
                    if (not self._mismatch_active and
                            self._mismatch_frames >= self._MISMATCH_TRIGGER):
                        self._mismatch_active = True
                        bad_entry = {
                            'rep':      len(self.rep_history) + 1,
                            'score':    penalty_score,   # 30–45 range
                            'error':    penalty_msg,
                            'mismatch': True,
                        }
                        self.rep_history.append(bad_entry)
                    elif self._mismatch_active:
                        # Update the last entry's score to current penalty (refresh)
                        if self.rep_history and self.rep_history[-1].get('mismatch'):
                            self.rep_history[-1]['score'] = penalty_score
                else:
                    # Correct exercise restored — close the mismatch bout
                    self._mismatch_active = False
                    self._mismatch_frames = 0
                    # ── Normal rep logging ────────────────────────────────────
                    if self.result.get('rep_triggered'):
                        entry = {
                            'rep':      self.result['count'],
                            'score':    self.result['form_score'],
                            'error':    self.result.get('last_error'),
                            'mismatch': False,
                        }
                        self.rep_history.append(entry)
            else:
                self.detected_exercise = None
                self.result = {**self.result,
                               'feedback': 'No pose — step back & stand tall',
                               'rep_triggered': False}

            draw_hud(frame_bgr, self.result, self.exercise, mismatch_ex)
            return frame_bgr

    def reset(self):
        with self.lock:
            self.analyzer.rc.reset()
            self.rep_history      = []
            self._mismatch_active = False
            self._mismatch_frames = 0
            self.result = {'count': 0, 'stage': '', 'feedback': 'Ready!',
                           'angle': 0, 'form_score': 0,
                           'last_error': None, 'rep_triggered': False}
            self.detected_exercise = None


# ── Session state ─────────────────────────────────────────────────────────────
if 'gym_state' not in st.session_state:
    st.session_state.gym_state = GymState()
gym = st.session_state.gym_state


# ── HTML render helpers ───────────────────────────────────────────────────────
def _score_color_hex(s):
    if s >= 75: return '#00e5a0'
    if s >= 50: return '#ffce3a'
    return '#ff5c3a'

def _score_tag(s):
    if s >= 75: return '<span class="hist-tag tag-good">Good</span>'
    if s >= 50: return '<span class="hist-tag tag-ok">Fair</span>'
    return '<span class="hist-tag tag-poor">Poor</span>'

def render_metric_row(count, stage, angle, score):
    stage_str = (stage or 'READY').upper()
    stage_col = '#4e8ef7' if stage_str not in ('UP', 'READY') else '#00e5a0'
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{count}</div>
        <div class="metric-label">Reps</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-value" style="color:{stage_col};font-size:1.5rem">{stage_str}</div>
        <div class="metric-label">Stage</div>
      </div>
      <div class="metric-card orange">
        <div class="metric-value">{int(angle)}°</div>
        <div class="metric-label">Angle</div>
      </div>
      <div class="metric-card yellow">
        <div class="metric-value" style="color:{_score_color_hex(score)}">{score}</div>
        <div class="metric-label">Form</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_feedback(fb):
    if fb:
        st.markdown(f"""
        <div class="feedback-card">
          <div class="fb-label">Feedback</div>
          <div class="fb-msg">{fb}</div>
        </div>""", unsafe_allow_html=True)

def render_warning(detected, selected):
    if detected and detected != selected:
        st.markdown(f"""
        <div class="warn-mismatch">
          ⚠️ &nbsp;<strong>Incorrect exercise selected.</strong><br>
          Pose detected as <strong>{detected}</strong>, but <strong>{selected}</strong> is active.
          Please switch the exercise dropdown or adjust your body position.
        </div>""", unsafe_allow_html=True)
    elif detected:
        st.markdown(f"""
        <div class="warn-ok">
          ✓ &nbsp;Pose matches <strong>{selected}</strong> — tracking active
        </div>""", unsafe_allow_html=True)

def render_history_panel(rep_history):
    real_reps   = [r for r in rep_history if not r.get('mismatch')]
    mis_events  = [r for r in rep_history if r.get('mismatch')]
    form_errors = [r for r in real_reps if r.get('error')]
    total_all   = len(rep_history)
    total_valid = len(real_reps)

    # Session score = sum of ALL scores / ALL events (mismatch scores=0 penalise avg)
    session_score = (
        int(sum(r['score'] for r in rep_history) / total_all)
        if total_all else 0
    )
    valid_avg = (
        int(sum(r['score'] for r in real_reps) / total_valid)
        if total_valid else 0
    )
    sc_col  = _score_color_hex(session_score) if total_all else '#6b7a92'
    avg_col = _score_color_hex(valid_avg)     if total_valid else '#6b7a92'

    # Score breakdown string e.g. "0 + 0 + 89 = 29"
    breakdown = ' + '.join(str(r['score']) for r in rep_history) if rep_history else '0'

    st.markdown(f"""
    <div class="session-summary">
      <div class="sess-row">
        <span class="sess-label">Session score</span>
        <span class="sess-val" style="color:{sc_col};font-size:22px">{session_score if total_all else '—'}</span>
      </div>
      <div style="font-size:10px;color:#6b7a92;font-family:var(--mono);margin:-2px 0 8px;word-break:break-all">
        ({breakdown}) / {total_all if total_all else 1} = {session_score}
      </div>
      <div style="height:1px;background:rgba(255,255,255,0.06);margin-bottom:8px"></div>
      <div class="sess-row">
        <span class="sess-label" style="font-size:10px">Valid reps</span>
        <span class="sess-val" style="color:#00e5a0;font-size:14px">{total_valid}</span>
      </div>
      <div class="sess-row">
        <span class="sess-label" style="font-size:10px">Valid avg form</span>
        <span class="sess-val" style="color:{avg_col};font-size:14px">{valid_avg if total_valid else '—'}</span>
      </div>
      <div class="sess-row">
        <span class="sess-label" style="font-size:10px">Form errors</span>
        <span class="sess-val" style="color:#ffce3a;font-size:14px">{len(form_errors)}</span>
      </div>
      <div class="sess-row">
        <span class="sess-label" style="font-size:10px">Wrong exercise</span>
        <span class="sess-val" style="color:#ff5c3a;font-size:14px">{len(mis_events)}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Rep-by-rep history (valid reps only) ────────────────────
    st.markdown('<div class="hist-header" style="margin-top:12px">Rep-by-rep history</div>',
                unsafe_allow_html=True)

    if not real_reps:
        st.markdown('<p style="color:var(--muted);font-size:13px;padding:6px 0 4px">No valid reps yet — start exercising.</p>',
                    unsafe_allow_html=True)
    else:
        for entry in reversed(real_reps[-20:]):
            sc      = entry['score']
            err     = entry.get('error')
            col     = _score_color_hex(sc)
            tag     = _score_tag(sc)
            border  = 'border-left:3px solid #ffce3a;' if err else ''
            err_html = f'<span class="hist-error-note">↳ {err}</span>' if err else ''
            st.markdown(f"""
            <div class="hist-item" style="{border}">
              <span class="hist-rep-num">R{entry['rep']}</span>
              <div style="flex:1">
                <div class="hist-bar-bg">
                  <div class="hist-bar-fill" style="width:{sc}%;background:{col}"></div>
                </div>
                {err_html}
              </div>
              <span class="hist-score" style="color:{col}">{sc}</span>
              {tag}
            </div>""", unsafe_allow_html=True)

    # ── Section 2: Exercise errors (mismatch events) ────────────────────────
    st.markdown('<div class="hist-header" style="margin-top:14px">Exercise errors</div>',
                unsafe_allow_html=True)

    if not mis_events:
        st.markdown('<p style="color:var(--muted);font-size:13px;padding:6px 0 4px">No wrong-exercise events — great discipline!</p>',
                    unsafe_allow_html=True)
    else:
        for i, entry in enumerate(reversed(mis_events[-20:]), 1):
            raw_msg = entry.get('error', 'Wrong exercise detected')
            # Strip any legacy "— 0 pts" suffix from old entries
            err_msg = raw_msg.replace(' — 0 pts', '').replace('— 0 pts', '').strip()
            sc_entry = entry.get('score', 35)
            if sc_entry == 0:
                sc_entry = 35   # upgrade legacy zero entries to fair penalty
            sc_entry_col = _score_color_hex(sc_entry)
            st.markdown(f"""
            <div class="hist-item" style="border-left:3px solid #ff5c3a;">
              <span class="hist-rep-num" style="color:#ff5c3a">✕{i}</span>
              <div style="flex:1">
                <div style="font-size:11px;color:#ff9688;line-height:1.45">{err_msg}</div>
                <div style="font-size:10px;color:#6b7a92;margin-top:2px">Score: {sc_entry} pts — penalises session avg</div>
              </div>
              <span class="hist-score" style="color:{sc_entry_col}">{sc_entry}</span>
              <span class="hist-tag" style="background:rgba(255,92,58,0.15);color:#ff5c3a;border:1px solid rgba(255,92,58,0.4)">Wrong</span>
            </div>""", unsafe_allow_html=True)


def render_end_button(session_score):
    _url = f"http://localhost/movera/patient/patient-plan.php?score={session_score}"
    st.markdown(f"""
    <div class="end-btn-wrap">
      <a href="{_url}" target="_blank" rel="noopener noreferrer">
        End Session &nbsp;·&nbsp; Score: {session_score}
      </a>
    </div>""", unsafe_allow_html=True)


def _calc_session_score(rep_history):
    """Total score across all events. Wrong-exercise entries count as 0."""
    if not rep_history:
        return 0
    return int(sum(r['score'] for r in rep_history) / len(rep_history))


def render_stats_panel():
    res   = gym.result
    cnt   = res.get('count', 0)
    stage = res.get('stage', '')
    ang   = float(res.get('angle') or 0)
    fb    = res.get('feedback', '')
    sc    = res.get('form_score', 0)
    det   = gym.detected_exercise

    render_warning(det, gym.exercise)
    render_metric_row(cnt, stage, ang, sc)
    render_feedback(fb)
    render_history_panel(gym.rep_history)
    session_score = _calc_session_score(gym.rep_history)
    render_end_button(session_score)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Gym Tracker")
    st.markdown("---")
    exercise = st.selectbox("Exercise", list(ANALYZERS.keys()))
    st.markdown("---")
    mode = st.radio("Mode", ["📹 Webcam (Real-Time)", "📁 Upload Video"],
                    label_visibility="collapsed")
    st.markdown("---")
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    mirror        = st.checkbox("Mirror Webcam",  value=True)
    st.markdown("---")
    if st.button("Reset Counter", use_container_width=True):
        gym.reset()
        st.rerun()
    st.markdown("---")
    st.markdown("""
    <div class="ex-chips">
      <span class="ex-chip">Squat</span>
      <span class="ex-chip">Push-Up</span>
      <span class="ex-chip">Pull-Up</span>
      <span class="ex-chip">Jump Jack</span>
      <span class="ex-chip">Rus. Twist</span>
    </div>
    """, unsafe_allow_html=True)

gym.set_exercise(exercise)
gym.show_skeleton = show_skeleton
gym.mirror        = mirror


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# ⚡ AI Gym Tracker")
st.markdown('<p class="subtitle">MediaPipe Heavy &nbsp;·&nbsp; Real-Time Pose &nbsp;·&nbsp; Form Scoring</p>',
            unsafe_allow_html=True)

col_vid, col_stats = st.columns([3, 1.6])


# ── Webcam mode ───────────────────────────────────────────────────────────────
if mode.startswith("📹"):
    with col_vid:
        st.markdown('<p style="font-family:var(--cond, sans-serif);font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#6b7a92;margin-bottom:6px">📹 Live Webcam — Real-Time Tracking</p>',
                    unsafe_allow_html=True)
        st.info("Click **START** → allow camera access → begin exercising!")

    def video_frame_callback(frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if gym.mirror:
            img = cv2.flip(img, 1)
        try:
            img = gym.process_frame(img)
        except Exception as e:
            cv2.putText(img, f"Err: {str(e)[:40]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return VideoFrame.from_ndarray(img, format="bgr24")

    with col_vid:
        ctx = webrtc_streamer(
            key=f"gym-{exercise}",
            mode=WebRtcMode.SENDRECV,
            server_rtc_configuration=SERVER_RTC_CONFIG,
            frontend_rtc_configuration=FRONTEND_RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

    with col_stats:
        render_stats_panel()

    if ctx.state.playing:
        import time
        while ctx.state.playing:
            with col_stats:
                render_stats_panel()
            time.sleep(0.25)


# ── Video upload mode ─────────────────────────────────────────────────────────
else:
    with col_vid:
        st.markdown('<p style="font-family:var(--cond,sans-serif);font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#6b7a92;margin-bottom:6px">📁 Upload Video for Analysis</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("MP4 / AVI / MOV / MKV",
                                    type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            st.video(uploaded)
            if st.button("Analyze Video", type="primary", use_container_width=True):
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python.vision import (
                    PoseLandmarker, PoseLandmarkerOptions)
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                    VisionTaskRunningMode)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name

                opts = PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionTaskRunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                lm  = PoseLandmarker.create_from_options(opts)
                cap = cv2.VideoCapture(tmp_path)
                vw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                analyzer          = ANALYZERS[exercise]()
                rep_log           = []   # {rep, score, error}
                mis_active        = False   # debounce: is a mismatch bout ongoing?
                mis_frames        = 0       # consecutive frames of mismatch
                MIS_TRIGGER       = 15      # frames before logging one event
                prog      = st.progress(0, "Processing…")
                preview   = st.empty()
                last_res  = {'count': 0, 'stage': '', 'feedback': '',
                             'angle': 0, 'form_score': 0,
                             'last_error': None, 'rep_triggered': False}
                fidx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    fidx += 1
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    det    = lm.detect_for_video(mp_img, int(fidx / fps * 1000))

                    mismatch_ex = None
                    if det.pose_landmarks:
                        lms = det.pose_landmarks[0]
                        guess = _guess_exercise(lms, vw, vh)
                        if guess and guess != exercise:
                            mismatch_ex = guess
                        if show_skeleton:
                            draw_skeleton(frame, lms, vw, vh)
                        last_res = analyzer.analyze(lms, vw, vh)

                        # ── Mismatch penalty (video) ─────────────────────────
                        if mismatch_ex:
                            mis_frames += 1
                            if last_res.get('rep_triggered'):
                                analyzer.rc.count = max(0, analyzer.rc.count - 1)
                                if analyzer.rc._angle_scores:
                                    analyzer.rc._angle_scores.pop()
                            import random
                            penalty_score = random.randint(30, 45)
                            penalty_msg = (
                                f"Wrong exercise! Doing {mismatch_ex} "
                                f"but '{exercise}' is selected"
                            )
                            last_res = {
                                **last_res,
                                'count':         analyzer.rc.count,
                                'form_score':    penalty_score,
                                'feedback':      penalty_msg,
                                'last_error':    penalty_msg,
                                'rep_triggered': False,
                            }
                            if not mis_active and mis_frames >= MIS_TRIGGER:
                                mis_active = True
                                rep_log.append({
                                    'rep':      len(rep_log) + 1,
                                    'score':    penalty_score,   # 30–45 range
                                    'error':    penalty_msg,
                                    'mismatch': True,
                                })
                            elif mis_active and rep_log and rep_log[-1].get('mismatch'):
                                rep_log[-1]['score'] = penalty_score
                        else:
                            mis_active = False
                            mis_frames = 0
                            # ── Normal rep logging ────────────────────────────
                            if last_res.get('rep_triggered'):
                                rep_log.append({
                                    'rep':      last_res['count'],
                                    'score':    last_res['form_score'],
                                    'error':    last_res.get('last_error'),
                                    'mismatch': False,
                                })
                    else:
                        last_res = {**last_res, 'feedback': 'No pose detected',
                                    'rep_triggered': False}

                    draw_hud(frame, last_res, exercise, mismatch_ex)

                    if fidx % 30 == 0 or fidx == tot:
                        preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                      use_container_width=True)
                        prog.progress(min(fidx / max(tot, 1), 1.0),
                                      text=f"Frame {fidx}/{tot} — Reps: {last_res['count']}")

                cap.release(); lm.close()
                os.unlink(tmp_path); prog.empty()
                st.success(f"✅ Done! **{last_res['count']} reps** detected in {fidx} frames.")

                # Push results to gym state for the stats panel
                gym.result      = last_res
                gym.rep_history = rep_log

                with col_stats:
                    render_stats_panel()

                hist = list(analyzer.rc.history)
                if hist:
                    import pandas as pd
                    st.markdown(
                        '<p style="font-family:var(--cond,sans-serif);font-size:12px;'
                        'letter-spacing:2px;text-transform:uppercase;color:#6b7a92;'
                        'margin:16px 0 6px">📈 Angle History</p>',
                        unsafe_allow_html=True)
                    st.line_chart(pd.DataFrame({'angle': hist}), color="#00e5a0")

    with col_stats:
        render_stats_panel()
