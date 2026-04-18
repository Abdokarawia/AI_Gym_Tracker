"""
AI Gym Tracker — Streamlit Real-Time App
Uses streamlit-webrtc for true browser-side web cam (works on Streamlit Cloud)
Supports: Squat · Push-Up · Pull-Up · Jumping Jack · Russian Twist

NEW in this version:
─────────────────────────────────────────────────────────────────────────────
1. Wrong-exercise detection: heuristics compare the current movement signature
   (which body part is moving, angle magnitudes, dominant motion axis) against
   the selected exercise, and warn the user when a mismatch is detected.

2. Exercise error / warning history panel in the sidebar: every wrong-exercise
   warning and form-feedback error is time-stamped and stored in session state,
   then rendered as a scrollable log with colour-coded severity badges.
─────────────────────────────────────────────────────────────────────────────
"""

# ── Asyncio / aioice Python-3.12+ shutdown patch ─────────────────────────────
import asyncio.selector_events as _sel

_orig_fatal = _sel._SelectorDatagramTransport._fatal_error  # type: ignore[attr-defined]

def _patched_fatal_error(self, exc, message="Fatal error on transport"):
    if self._loop is None:
        return
    try:
        _orig_fatal(self, exc, message)
    except Exception:
        pass

_sel._SelectorDatagramTransport._fatal_error = _patched_fatal_error  # type: ignore[attr-defined]
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import cv2
import numpy as np
import math
import os
import tempfile
import threading
from collections import deque
from datetime import datetime
from av import VideoFrame
from aiortc import RTCConfiguration as AioRTCConfiguration, RTCIceServer
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(
    page_title="AI Gym Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap');

body, .stApp { background-color: #0d0f12; color: #e8eaf0; font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background-color: #13161c !important; border-right: 1px solid #1e2229; }
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #39ff7e !important; font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }
[data-testid="stSidebar"] hr { border-color: #1e2229 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label { color: #9ca3af !important; font-size:0.85rem; letter-spacing:0.5px; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #1a1d25 !important; border-color: #2a2f3a !important; color: #e8eaf0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #e8eaf0 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { color: #9ca3af !important; }
[data-testid="stSidebar"] .stButton > button { background-color: #1a1d25 !important; color: #e8eaf0 !important; border: 1px solid #2a2f3a !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] .stButton > button:hover { background-color: #22262f !important; border-color: #39ff7e !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: #39ff7e !important; color: #0d0f12 !important; border: none !important; font-weight: 700; }

h1 { color: #39ff7e !important; font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 3px !important; font-size: 2.4rem !important; }
h4 { color: #9ca3af !important; font-size: 0.9rem !important; letter-spacing: 1px; text-transform: uppercase; }

.metric-card { background: #13161c; border-radius: 12px; padding: 16px 20px; border: 1px solid #1e2229; text-align: center; margin-bottom: 10px; position: relative; overflow: hidden; }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg, #39ff7e, #00d4ff); }
.metric-value { font-size: 3rem; font-weight: 800; color: #39ff7e; line-height: 1.1; font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }
.metric-label { font-size: 0.72rem; color: #4b5563; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }
.feedback-box { background: #13161c; border-left: 3px solid #39ff7e; border-radius: 8px; padding: 10px 16px; margin-top: 10px; font-size: 0.95rem; color: #e8eaf0; }

/* Wrong exercise warning banner */
.wrong-ex-banner { background: linear-gradient(135deg, #2d1a00, #3d2200); border: 1px solid #f59e0b; border-radius: 10px; padding: 12px 18px; margin: 8px 0; font-size: 0.92rem; color: #fcd34d; animation: pulse-border 1.5s ease-in-out infinite; }
@keyframes pulse-border { 0%,100%{border-color:#f59e0b;} 50%{border-color:#fbbf24;box-shadow:0 0 12px rgba(251,191,36,0.3);} }

/* History log */
.history-log { max-height: 320px; overflow-y: auto; padding-right: 4px; }
.history-log::-webkit-scrollbar { width: 4px; }
.history-log::-webkit-scrollbar-track { background: #0d0f12; }
.history-log::-webkit-scrollbar-thumb { background: #2a2f3a; border-radius: 2px; }
.log-entry { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; margin-bottom: 5px; border-radius: 8px; background: #13161c; border: 1px solid #1e2229; font-size: 0.82rem; line-height: 1.4; }
.log-badge { flex-shrink: 0; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-top: 1px; }
.badge-wrong { background: #78350f; color: #fcd34d; }
.badge-form  { background: #1e1b4b; color: #a5b4fc; }
.badge-info  { background: #052e16; color: #6ee7b7; }
.log-time { color: #4b5563; font-size: 0.72rem; flex-shrink: 0; margin-top: 2px; }
.log-text { color: #d1d5db; flex: 1; }
.no-history { color: #374151; font-size: 0.82rem; font-style: italic; text-align: center; padding: 20px 0; }
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
    RTCIceServer(urls=[f"stun:stun.l.google.com:19302"]),
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
        self.up_thresh     = up_thresh
        self.down_thresh   = down_thresh
        self.higher_is_up  = higher_is_up
        self.count         = 0
        self.stage         = None
        self.history       = deque(maxlen=2000)
        self._rep_best     = None
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

# ── Wrong-exercise detector ───────────────────────────────────────────────────
class WrongExerciseDetector:
    """
    Lightweight heuristic detector.

    For each exercise we define which body region is expected to be most active
    and what angle ranges are characteristic.  We compare against the *other*
    exercises to see if a different pattern better explains the pose.

    Signatures return a confidence in [0,1] for how well the current pose
    fits each exercise.
    """
    WARN_FRAMES = 20   # consecutive mismatch frames before firing
    COOLDOWN    = 90   # frames between repeated warnings

    def __init__(self):
        self._mismatch_count = 0
        self._last_warning   = 0
        self._frame_idx      = 0

    def _scores(self, lms, w, h):
        """Return dict[exercise_name -> confidence 0-1]."""
        scores = {}

        def ang(a, b, c):
            return angle3(lm_px(lms, a, w, h),
                          lm_px(lms, b, w, h),
                          lm_px(lms, c, w, h))

        # ── Squat: deep knee bend, hips low ─────────────────────────────────
        knee_ang   = ang(LM['l_hip'], LM['l_knee'], LM['l_ankle'])
        hip_height = lm_px(lms, LM['l_hip'], w, h)[1] / h
        squat_sc   = max(0, 1 - abs(knee_ang - 90) / 90) * (0.4 + 0.6 * hip_height)
        scores['Squat'] = min(1.0, squat_sc)

        # ── Push-Up: elbow bent + near-horizontal torso ──────────────────────
        elbow_ang  = ang(LM['l_shoulder'], LM['l_elbow'], LM['l_wrist'])
        sh_y       = lm_px(lms, LM['l_shoulder'], w, h)[1] / h
        hip_y      = lm_px(lms, LM['l_hip'],      w, h)[1] / h
        torso_h    = 1 - abs(sh_y - hip_y)
        pushup_sc  = max(0, 1 - abs(elbow_ang - 90) / 90) * (0.3 + 0.7 * torso_h)
        scores['Push-Up'] = min(1.0, pushup_sc)

        # ── Pull-Up: elbows highly bent + wrists above shoulders ─────────────
        wr_y       = lm_px(lms, LM['l_wrist'],    w, h)[1] / h
        sh_y2      = lm_px(lms, LM['l_shoulder'], w, h)[1] / h
        above      = max(0, sh_y2 - wr_y)
        pullup_sc  = max(0, 1 - abs(elbow_ang - 60) / 80) * (0.3 + 0.7 * above * 4)
        scores['Pull-Up'] = min(1.0, pullup_sc)

        # ── Jumping Jack: arms raised laterally ─────────────────────────────
        arm_ang    = ang(LM['l_hip'], LM['l_shoulder'], LM['l_wrist'])
        jj_sc      = max(0, (arm_ang - 60) / 100)
        scores['Jumping Jack'] = min(1.0, jj_sc)

        # ── Russian Twist: lateral wrist offset from hip centre ───────────────
        lwr_x      = lm_px(lms, LM['l_wrist'], w, h)[0] / w
        rwr_x      = lm_px(lms, LM['r_wrist'], w, h)[0] / w
        rt_sc      = min(1.0, max(0, abs(lwr_x - rwr_x) - 0.05) * 2)
        scores['Russian Twist'] = rt_sc

        return scores

    def check(self, selected_exercise, lms, w, h):
        """
        Returns (is_wrong: bool, suspected: str, confidence: float).
        is_wrong fires only after WARN_FRAMES consecutive mismatches.
        """
        self._frame_idx += 1
        scores   = self._scores(lms, w, h)
        best_ex  = max(scores, key=scores.__getitem__)
        best_sc  = scores[best_ex]
        sel_sc   = scores.get(selected_exercise, 0)
        margin   = best_sc - sel_sc

        mismatch = (best_ex != selected_exercise
                    and margin > 0.30
                    and best_sc > 0.35)

        if mismatch:
            self._mismatch_count += 1
        else:
            self._mismatch_count = max(0, self._mismatch_count - 2)

        cooldown_ok = (self._frame_idx - self._last_warning) > self.COOLDOWN
        if self._mismatch_count >= self.WARN_FRAMES and cooldown_ok:
            self._last_warning   = self._frame_idx
            self._mismatch_count = 0
            return True, best_ex, round(best_sc, 2)

        return False, best_ex, round(best_sc, 2)

# ── Exercise analyzers ────────────────────────────────────────────────────────
class SquatAnalyzer:
    label = 'Squat'
    def __init__(self): self.rc = RepCounter(160, 90)
    def analyze(self, lms, w, h):
        hip   = lm_px(lms, LM['l_hip'],   w, h)
        knee  = lm_px(lms, LM['l_knee'],  w, h)
        ankle = lm_px(lms, LM['l_ankle'], w, h)
        ang = angle3(hip, knee, ankle)
        rep = self.rc.update(ang)
        if rep == 'rep' and self.rc._rep_best is not None:
            self.rc._angle_scores.append(_angle_score(self.rc._rep_best, 70, 160))
        fb = ('Go lower!'           if ang > 110 and self.rc.stage != 'down'
              else 'Good depth! 🔥' if ang < 90
              else 'Stand tall!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score())

class PushUpAnalyzer:
    label = 'Push-Up'
    def __init__(self): self.rc = RepCounter(155, 90)
    def analyze(self, lms, w, h):
        sh = lm_px(lms, LM['l_shoulder'], w, h)
        el = lm_px(lms, LM['l_elbow'],   w, h)
        wr = lm_px(lms, LM['l_wrist'],   w, h)
        ang = angle3(sh, el, wr)
        rep = self.rc.update(ang)
        if rep == 'rep' and self.rc._rep_best is not None:
            self.rc._angle_scores.append(_angle_score(self.rc._rep_best, 60, 155))
        fb = ('Lower chest!'        if ang > 130 and self.rc.stage != 'down'
              else 'Good depth! 🔥' if ang < 90
              else 'Push up!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score())

class PullUpAnalyzer:
    label = 'Pull-Up'
    def __init__(self): self.rc = RepCounter(50, 140, higher_is_up=False)
    def analyze(self, lms, w, h):
        sh = lm_px(lms, LM['l_shoulder'], w, h)
        el = lm_px(lms, LM['l_elbow'],   w, h)
        wr = lm_px(lms, LM['l_wrist'],   w, h)
        ang = angle3(sh, el, wr)
        rep = self.rc.update(ang)
        if rep == 'rep' and self.rc._rep_best is not None:
            self.rc._angle_scores.append(_angle_score(self.rc._rep_best, 30, 140))
        fb = ('Pull higher!'            if ang > 70 and self.rc.stage == 'up'
              else 'Chin over bar! 🔥'  if ang < 50
              else 'Lower slowly!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score())

class JumpingJackAnalyzer:
    label = 'Jumping Jack'
    def __init__(self): self.rc = RepCounter(130, 40); self._rep_max = 0
    def analyze(self, lms, w, h):
        lsh = lm_px(lms, LM['l_shoulder'], w, h)
        lhi = lm_px(lms, LM['l_hip'],     w, h)
        lwr = lm_px(lms, LM['l_wrist'],   w, h)
        ang = angle3(lhi, lsh, lwr)
        self._rep_max = max(self._rep_max, ang)
        rep = self.rc.update(ang)
        if rep == 'rep':
            self.rc._angle_scores.append(_angle_score(self._rep_max, 150, 40))
            self._rep_max = 0
        fb = 'Arms up!' if ang < 80 else ('Great! 🔥' if ang > 120 else 'Keep going!')
        return dict(angle=ang, stage=self.rc.stage, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score())

class RussianTwistAnalyzer:
    label = 'Russian Twist'
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
        if rot > 30 and side != self._last_side:
            self._last_side = side; self._touches += 1
            if self._touches % 2 == 0:
                self.rc.count += 1
                self.rc._angle_scores.append(_angle_score(self._touch_max, 80, 0))
                self._touch_max = 0
        self.rc.history.append(rot)
        fb = f'Twist {side}!' if rot < 20 else f'Good twist → {side}! 🔥'
        return dict(angle=rot, stage=side, count=self.rc.count,
                    feedback=fb, form_score=self.rc.angle_score())

ANALYZERS = {
    'Squat':         SquatAnalyzer,
    'Push-Up':       PushUpAnalyzer,
    'Pull-Up':       PullUpAnalyzer,
    'Jumping Jack':  JumpingJackAnalyzer,
    'Russian Twist': RussianTwistAnalyzer,
}

# ── Drawing helpers ───────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_DUPLEX

def draw_skeleton(frame, lms, w, h):
    for a, b in SKELETON_EDGES:
        cv2.line(frame, lm_px(lms, a, w, h), lm_px(lms, b, w, h),
                 (57, 255, 126), 2, cv2.LINE_AA)
    for i in range(33):
        pt = lm_px(lms, i, w, h)
        cv2.circle(frame, pt, 5, (74, 163, 22), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255),  1, cv2.LINE_AA)

def draw_wrong_ex_overlay(frame, suspected):
    """Amber warning bar at top of frame."""
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 48), (10, 90, 200), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    msg = f"WRONG EXERCISE? Looks like: {suspected.upper()}"
    tw, _ = cv2.getTextSize(msg, FONT, 0.55, 1)[0]
    cv2.putText(frame, msg, (W//2 - tw//2, 32),
                FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, msg, (W//2 - tw//2, 32),
                FONT, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

def draw_hud(frame, res, ex_title):
    H, W    = frame.shape[:2]
    count   = res.get('count', 0)
    stage   = res.get('stage', '') or ''
    feedback = res.get('feedback', '')
    angle_v  = float(res.get('angle') or 0)

    label = ex_title.upper()
    lw, _ = cv2.getTextSize(label, FONT, 0.65, 1)[0]
    cv2.rectangle(frame, (W//2-lw//2-12, 6), (W//2+lw//2+12, 38), (13,16,20), -1)
    cv2.putText(frame, label, (W//2-lw//2, 30), FONT, 0.65, (0,  0,  0),  3, cv2.LINE_AA)
    cv2.putText(frame, label, (W//2-lw//2, 30), FONT, 0.65, (57,255,126), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (8,48),  (138,178), (13,16,20), -1)
    cv2.rectangle(frame, (8,48),  (138,52),  (57,255,126), -1)
    cv2.putText(frame, 'REPS', (18,73), FONT, 0.42, (100,108,120), 1, cv2.LINE_AA)
    cstr = str(count)
    cw, _ = cv2.getTextSize(cstr, FONT, 2.6, 2)[0]
    cv2.putText(frame, cstr, (73-cw//2,158), FONT, 2.6, (0,  0,  0),  5, cv2.LINE_AA)
    cv2.putText(frame, cstr, (73-cw//2,158), FONT, 2.6, (57,255,126), 2, cv2.LINE_AA)

    sl = stage.upper() if stage else 'READY'
    sc = (57,255,126) if stage == 'up' else (60,100,235)
    cv2.rectangle(frame, (8,186), (138,212), sc, -1)
    sw, _ = cv2.getTextSize(sl, FONT, 0.42, 1)[0]
    cv2.putText(frame, sl, (73-sw//2,205), FONT, 0.42, (0,0,0), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (W-148,48), (W-8,178), (13,16,20), -1)
    cv2.rectangle(frame, (W-148,48), (W-8,52),  (235,99,37), -1)
    cv2.putText(frame, 'ANGLE', (W-138,73), FONT, 0.42, (100,108,120), 1, cv2.LINE_AA)
    astr = f'{int(angle_v)} deg'
    aw, _ = cv2.getTextSize(astr, FONT, 0.75, 1)[0]
    cv2.putText(frame, astr, (W-78-aw//2,130), FONT, 0.75, (0,  0,  0),  3, cv2.LINE_AA)
    cv2.putText(frame, astr, (W-78-aw//2,130), FONT, 0.75, (235,99,37),  1, cv2.LINE_AA)

    if feedback:
        fb   = feedback[:58]
        fw, fh = cv2.getTextSize(fb, FONT, 0.5, 1)[0]
        fy2 = H-12; fy1 = fy2-fh-16
        cv2.rectangle(frame, (W//2-fw//2-18,fy1), (W//2+fw//2+18,fy2), (13,16,20), -1)
        cv2.putText(frame, fb, (W//2-fw//2,fy2-7), FONT, 0.5, (0,  0,  0),  3, cv2.LINE_AA)
        cv2.putText(frame, fb, (W//2-fw//2,fy2-7), FONT, 0.5, (255,255,255),1, cv2.LINE_AA)

# ── Event history ─────────────────────────────────────────────────────────────
class EventHistory:
    """Thread-safe ring buffer for exercise events/warnings."""
    MAX = 100

    def __init__(self):
        self._lock   = threading.Lock()
        self._events = deque(maxlen=self.MAX)

    def add(self, kind: str, message: str):
        """kind: 'wrong' | 'form' | 'info'"""
        with self._lock:
            self._events.appendleft({
                'time':    datetime.now().strftime('%H:%M:%S'),
                'kind':    kind,
                'message': message,
            })

    def all(self):
        with self._lock:
            return list(self._events)

    def clear(self):
        with self._lock:
            self._events.clear()

# ── Shared gym state ──────────────────────────────────────────────────────────
# Bump _GYM_STATE_VERSION whenever GymState gains or removes attributes.
# The session-state guard below will rebuild the object automatically, which
# prevents AttributeError crashes when users have a live session during deploy.
_GYM_STATE_VERSION = 2

class GymState:
    def __init__(self):
        self._version      = _GYM_STATE_VERSION   # bump to force refresh after deploys
        self.lock          = threading.Lock()
        self.result        = {'count':0,'stage':'','feedback':'Get in position!',
                              'angle':0,'form_score':0}
        self.exercise      = 'Squat'
        self.show_skeleton = True
        self.mirror        = True
        self.analyzer      = SquatAnalyzer()
        self.wrong_ex_det  = WrongExerciseDetector()
        self.wrong_ex_flag = False
        self.wrong_ex_name = ''
        self._mp           = None
        self._landmarker   = None
        self.history       = EventHistory()
        self._last_feedback = ''

    def set_exercise(self, ex):
        with self.lock:
            if ex != self.exercise:
                self.exercise = ex
                self.analyzer = ANALYZERS[ex]()
                self.wrong_ex_det  = WrongExerciseDetector()
                self.wrong_ex_flag = False
                self.wrong_ex_name = ''
                self.result   = {'count':0,'stage':'','feedback':'Get in position!',
                                 'angle':0,'form_score':0}
                self._close_landmarker()
                self.history.add('info', f'Switched to {ex}')

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
            if det.pose_landmarks:
                lms = det.pose_landmarks[0]
                if self.show_skeleton:
                    draw_skeleton(frame_bgr, lms, W, H)

                # ── Wrong-exercise check ──────────────────────────────────
                is_wrong, suspected, conf = self.wrong_ex_det.check(
                    self.exercise, lms, W, H)
                if is_wrong:
                    self.wrong_ex_flag = True
                    self.wrong_ex_name = suspected
                    self.history.add(
                        'wrong',
                        f'Detected {suspected} (conf {conf:.0%}) — '
                        f'selected: {self.exercise}')
                    draw_wrong_ex_overlay(frame_bgr, suspected)
                else:
                    self.wrong_ex_flag = False

                # ── Pose analysis ─────────────────────────────────────────
                self.result = self.analyzer.analyze(lms, W, H)

                # Log corrective form cues (deduplicated)
                fb = self.result.get('feedback', '')
                if fb and fb != self._last_feedback:
                    self._last_feedback = fb
                    if any(kw in fb.lower() for kw in
                           ['lower', 'higher', 'go', 'pull', 'push',
                            'twist', 'arms', 'chin', 'chest']):
                        self.history.add('form', fb)
            else:
                self.result = {**self.result,
                               'feedback': 'No pose — step back & stand tall'}
            draw_hud(frame_bgr, self.result, self.exercise)
            return frame_bgr

# ── Session state ─────────────────────────────────────────────────────────────
# Version stamp — bump this any time GymState gains new attributes.
# If the cached object is from an older version it gets replaced cleanly,
# which fixes the "AttributeError: 'GymState' has no attribute 'X'" crash
# that occurs on Streamlit Cloud after a hot-reload with a live session.

def _gym_state_is_fresh(obj) -> bool:
    """Return True only if obj was built by the current GymState definition."""
    return (isinstance(obj, GymState)
            and getattr(obj, '_version', 0) == _GYM_STATE_VERSION)

if not _gym_state_is_fresh(st.session_state.get('gym_state')):
    st.session_state.gym_state = GymState()

gym = st.session_state.gym_state

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💪 AI Gym Tracker")
    st.markdown("---")
    exercise = st.selectbox("Exercise", list(ANALYZERS.keys()))
    st.markdown("---")
    mode = st.radio("Mode", ["📹 Webcam (Real-Time)", "📁 Upload Video"],
                    label_visibility="collapsed")
    st.markdown("---")
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    mirror        = st.checkbox("Mirror Webcam",  value=True)
    st.markdown("---")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Reset", use_container_width=True):
            with gym.lock:
                gym.analyzer.rc.reset()
                gym.result = {'count':0,'stage':'','feedback':'Ready!',
                              'angle':0,'form_score':0}
            st.rerun()
    with col_r2:
        if st.button("🗑 Clear Log", use_container_width=True):
            gym.history.clear()
            st.rerun()

    st.markdown("---")

    # ── Event History Panel ───────────────────────────────────────────────────
    st.markdown("### 📋 Exercise Log")
    events = gym.history.all()
    if events:
        badge_map = {
            'wrong': ('WRONG EX', 'badge-wrong'),
            'form':  ('FORM',     'badge-form'),
            'info':  ('INFO',     'badge-info'),
        }
        html_parts = ['<div class="history-log">']
        for ev in events:
            label_txt, badge_cls = badge_map.get(ev['kind'], ('LOG', 'badge-info'))
            html_parts.append(
                f'<div class="log-entry">'
                f'<span class="log-badge {badge_cls}">{label_txt}</span>'
                f'<span class="log-time">{ev["time"]}</span>'
                f'<span class="log-text">{ev["message"]}</span>'
                f'</div>'
            )
        html_parts.append('</div>')
        st.markdown(''.join(html_parts), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="no-history">No events yet — start exercising!</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Exercises:**\n🦵 Squat · 💪 Push-Up\n🏋️ Pull-Up · 🙆 Jumping Jack\n🔄 Russian Twist")

gym.set_exercise(exercise)
gym.show_skeleton = show_skeleton
gym.mirror        = mirror

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("💪 AI GYM TRACKER — REAL-TIME")
st.caption("MediaPipe Pose Estimation · Select exercise · Allow camera when prompted")

col_vid, col_stats = st.columns([3, 1])

with col_stats:
    wrong_ex_ph = st.empty()
    rep_ph      = st.empty()
    stage_ph    = st.empty()
    angle_ph    = st.empty()
    fb_ph       = st.empty()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    end_btn_ph  = st.empty()

def render_stats():
    res   = gym.result
    cnt   = res.get('count', 0)
    s     = (res.get('stage', '') or 'READY').upper()
    ang   = int(res.get('angle') or 0)
    fb    = res.get('feedback', '')
    sc100 = res.get('form_score', 0)
    sc    = '#39ff7e' if s in ('UP', 'READY') else '#3b82f6'

    # Wrong exercise banner
    if gym.wrong_ex_flag:
        wrong_ex_ph.markdown(
            f'<div class="wrong-ex-banner">'
            f'⚠️ Wrong exercise detected!<br>'
            f'<strong>Looks like: {gym.wrong_ex_name}</strong><br>'
            f'<small>Switch the selector or adjust your position</small>'
            f'</div>',
            unsafe_allow_html=True)
    else:
        wrong_ex_ph.empty()

    rep_ph.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{cnt}</div>'
        f'<div class="metric-label">Reps</div></div>',
        unsafe_allow_html=True)
    stage_ph.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="font-size:1.8rem;color:{sc};">{s}</div>'
        f'<div class="metric-label">Stage</div></div>',
        unsafe_allow_html=True)
    angle_ph.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:#00d4ff;">{ang}°</div>'
        f'<div class="metric-label">Angle</div></div>',
        unsafe_allow_html=True)
    if fb:
        fb_ph.markdown(f'<div class="feedback-box">💬 {fb}</div>',
                       unsafe_allow_html=True)
    _url = f"http://localhost/movera/patient/patient-plan.php?score={sc100}"
    end_btn_ph.markdown(
        f'<a href="{_url}" target="_blank" rel="noopener noreferrer"'
        f' style="display:block;width:100%;padding:10px 0;background:#39ff7e;color:#0d0f12;'
        f'text-align:center;border-radius:10px;font-weight:700;font-size:1rem;'
        f'text-decoration:none;margin-top:8px;">'
        f'End Exercise (Score: {sc100})</a>',
        unsafe_allow_html=True)

render_stats()

# ── Webcam mode ───────────────────────────────────────────────────────────────
if mode.startswith("📹"):
    with col_vid:
        st.markdown("#### 📹 Live Webcam — Real-Time Tracking")
        st.info("Click **START** → allow camera access → begin exercising!")

    def video_frame_callback(frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if gym.mirror:
            img = cv2.flip(img, 1)
        try:
            img = gym.process_frame(img)
        except Exception as e:
            cv2.putText(img, f"Err:{str(e)[:35]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return VideoFrame.from_ndarray(img, format="bgr24")

    with col_vid:
        ctx = webrtc_streamer(
            key=f"gym-{exercise}",
            mode=WebRtcMode.SENDRECV,
            server_rtc_configuration=SERVER_RTC_CONFIG,
            frontend_rtc_configuration=FRONTEND_RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"width": 640, "height": 480},
                                      "audio": False},
            async_processing=True,
            translations={
                "button.start":                        "▶ Start Camera",
                "button.stop":                         "■ Stop",
                "message.requesting_camera":           "Requesting camera access…",
                "message.camera_starting":             "Camera starting…",
                "message.media_devices_not_found":     "No camera found.",
                "message.media_devices_access_denied": "Camera access denied — check browser settings.",
            },
        )

    if ctx.state.playing:
        import time
        while ctx.state.playing:
            render_stats()
            time.sleep(0.25)

# ── Video upload mode ─────────────────────────────────────────────────────────
else:
    with col_vid:
        st.markdown("#### 📁 Upload Video for Analysis")
        uploaded = st.file_uploader("MP4 / AVI / MOV",
                                    type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            st.video(uploaded)
            if st.button("🧠 Analyze Video", type="primary",
                         use_container_width=True):
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python.vision import (
                    PoseLandmarker, PoseLandmarkerOptions)
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                    VisionTaskRunningMode)

                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=".mp4") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name

                opts = PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=MODEL_PATH),
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

                analyzer  = ANALYZERS[exercise]()
                det_video = WrongExerciseDetector()
                prog      = st.progress(0, "Processing…")
                preview   = st.empty()
                last_res  = {'count':0,'stage':'','feedback':'','angle':0}
                fidx      = 0
                gym.history.add('info', f'Video analysis started: {exercise}')

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    fidx += 1
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=rgb)
                    det = lm.detect_for_video(mp_img, int(fidx / fps * 1000))
                    if det.pose_landmarks:
                        lms = det.pose_landmarks[0]
                        if show_skeleton:
                            draw_skeleton(frame, lms, vw, vh)
                        is_wrong, suspected, conf = det_video.check(
                            exercise, lms, vw, vh)
                        if is_wrong:
                            draw_wrong_ex_overlay(frame, suspected)
                            gym.history.add(
                                'wrong',
                                f'[Frame {fidx}] Detected {suspected} '
                                f'(conf {conf:.0%})')
                        last_res = analyzer.analyze(lms, vw, vh)
                    else:
                        last_res = {**last_res, 'feedback': 'No pose detected'}
                    draw_hud(frame, last_res, exercise)
                    if fidx % 30 == 0 or fidx == tot:
                        preview.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            use_container_width=True)
                        prog.progress(
                            min(fidx / max(tot, 1), 1.0),
                            text=f"Frame {fidx}/{tot} — Reps: {last_res['count']}")

                cap.release(); lm.close()
                os.unlink(tmp_path); prog.empty()
                gym.history.add('info', f'Video done — {last_res["count"]} reps')
                st.success(
                    f"✅ Done! **{last_res['count']} reps** in {fidx} frames.")
                gym.result = last_res
                render_stats()

                hist = list(analyzer.rc.history)
                if hist:
                    import pandas as pd
                    st.markdown("#### 📈 Angle History")
                    st.line_chart(pd.DataFrame({'angle': hist}),
                                  color="#39ff7e")
