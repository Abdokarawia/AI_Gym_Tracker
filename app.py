"""
AI Gym Tracker — Streamlit Cloud Compatible
============================================
• MediaPipe JS (WASM) runs pose detection entirely in the browser
• No frames sent to Python — zero network latency, works everywhere
• Python only manages rep counts / scores passed back via st.query_params
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import math
import os
import base64
import threading
import time
import tempfile
from collections import deque
from datetime import datetime

st.set_page_config(
    page_title="AI Gym Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap');
body,.stApp{background:#0d0f12;color:#e8eaf0;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#13161c!important;border-right:1px solid #1e2229;}
[data-testid="stSidebar"] *{color:#e8eaf0!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#39ff7e!important;font-family:'Bebas Neue',sans-serif;letter-spacing:2px;}
[data-testid="stSidebar"] hr{border-color:#1e2229!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#1a1d25!important;border-color:#2a2f3a!important;color:#e8eaf0!important;}
[data-testid="stSidebar"] .stButton>button{background:#1a1d25!important;color:#e8eaf0!important;border:1px solid #2a2f3a!important;border-radius:8px!important;}
[data-testid="stSidebar"] .stButton>button:hover{background:#22262f!important;border-color:#39ff7e!important;}
h1{color:#39ff7e!important;font-family:'Bebas Neue',sans-serif!important;letter-spacing:3px!important;font-size:2.4rem!important;}
.metric-card{background:#13161c;border-radius:12px;padding:16px 20px;border:1px solid #1e2229;text-align:center;margin-bottom:10px;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#39ff7e,#00d4ff);}
.metric-value{font-size:3rem;font-weight:800;color:#39ff7e;line-height:1.1;font-family:'Bebas Neue',sans-serif;letter-spacing:2px;}
.metric-label{font-size:.72rem;color:#4b5563;letter-spacing:2px;text-transform:uppercase;margin-top:4px;}
.feedback-box{background:#13161c;border-left:3px solid #39ff7e;border-radius:8px;padding:10px 16px;margin-top:10px;font-size:.95rem;color:#e8eaf0;}
.wrong-ex-banner{background:linear-gradient(135deg,#2d1a00,#3d2200);border:1px solid #f59e0b;border-radius:10px;padding:12px 18px;margin:8px 0;font-size:.92rem;color:#fcd34d;}
.history-log{max-height:320px;overflow-y:auto;}
.log-entry{display:flex;gap:10px;align-items:flex-start;padding:8px 10px;margin-bottom:5px;border-radius:8px;background:#13161c;border:1px solid #1e2229;font-size:.82rem;line-height:1.4;}
.log-badge{flex-shrink:0;padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;}
.badge-wrong{background:#78350f;color:#fcd34d;}
.badge-form{background:#1e1b4b;color:#a5b4fc;}
.badge-info{background:#052e16;color:#6ee7b7;}
.log-time{color:#4b5563;font-size:.72rem;flex-shrink:0;}
.log-text{color:#d1d5db;flex:1;}
.no-history{color:#374151;font-size:.82rem;font-style:italic;text-align:center;padding:20px 0;}
</style>
""", unsafe_allow_html=True)

# ── MediaPipe model (for video upload mode only) ──────────────────────────────
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

# ── Geometry helpers (used in upload mode) ────────────────────────────────────
LM = {
    'l_shoulder':11,'r_shoulder':12,'l_elbow':13,'r_elbow':14,
    'l_wrist':15,'r_wrist':16,'l_hip':23,'r_hip':24,
    'l_knee':25,'r_knee':26,'l_ankle':27,'r_ankle':28,
}
SKELETON_EDGES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
]
FONT = cv2.FONT_HERSHEY_DUPLEX

def lm_px(lms,idx,w,h): lm=lms[idx]; return int(lm.x*w),int(lm.y*h)
def angle3(a,b,c):
    a,b,c=np.array(a,float),np.array(b,float),np.array(c,float)
    ba,bc=a-b,c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    return math.degrees(math.acos(np.clip(cos,-1,1)))
def _angle_score(best,ideal,worst):
    if ideal<worst: return max(0,min(100,int((worst-best)/(worst-ideal)*100)))
    return max(0,min(100,int((best-worst)/(ideal-worst)*100)))

class RepCounter:
    def __init__(self,up,down,higher_is_up=True):
        self.up_thresh=up;self.down_thresh=down;self.higher_is_up=higher_is_up
        self.count=0;self.stage=None;self.history=deque(maxlen=2000)
        self._rep_best=None;self._angle_scores=[]
    def update(self,v):
        self.history.append(v)
        if self._rep_best is None: self._rep_best=v
        if self.higher_is_up:
            self._rep_best=min(self._rep_best,v)
            if v>self.up_thresh: self.stage='up';self._rep_best=None
            elif v<self.down_thresh and self.stage=='up':
                self.stage='down';self.count+=1;return 'rep'
        else:
            self._rep_best=max(self._rep_best,v)
            if v<self.up_thresh: self.stage='up';self._rep_best=None
            elif v>self.down_thresh and self.stage=='up':
                self.stage='down';self.count+=1;return 'rep'
        return ''
    def angle_score(self):
        return int(sum(self._angle_scores)/len(self._angle_scores)) if self._angle_scores else 0
    def reset(self):
        self.count=0;self.stage=None;self.history.clear();self._rep_best=None;self._angle_scores=[]

class SquatAnalyzer:
    label='Squat'
    def __init__(self): self.rc=RepCounter(160,90)
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_hip'],w,h),lm_px(lms,LM['l_knee'],w,h),lm_px(lms,LM['l_ankle'],w,h))
        rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None: self.rc._angle_scores.append(_angle_score(self.rc._rep_best,70,160))
        fb='Go lower!' if ang>110 and self.rc.stage!='down' else('Good depth! 🔥' if ang<90 else 'Stand tall!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class PushUpAnalyzer:
    label='Push-Up'
    def __init__(self): self.rc=RepCounter(155,90)
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_shoulder'],w,h),lm_px(lms,LM['l_elbow'],w,h),lm_px(lms,LM['l_wrist'],w,h))
        rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None: self.rc._angle_scores.append(_angle_score(self.rc._rep_best,60,155))
        fb='Lower chest!' if ang>130 and self.rc.stage!='down' else('Good depth! 🔥' if ang<90 else 'Push up!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class PullUpAnalyzer:
    label='Pull-Up'
    def __init__(self): self.rc=RepCounter(50,140,higher_is_up=False)
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_shoulder'],w,h),lm_px(lms,LM['l_elbow'],w,h),lm_px(lms,LM['l_wrist'],w,h))
        rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None: self.rc._angle_scores.append(_angle_score(self.rc._rep_best,30,140))
        fb='Pull higher!' if ang>70 and self.rc.stage=='up' else('Chin over bar! 🔥' if ang<50 else 'Lower slowly!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class JumpingJackAnalyzer:
    label='Jumping Jack'
    def __init__(self): self.rc=RepCounter(130,40); self._rm=0
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_hip'],w,h),lm_px(lms,LM['l_shoulder'],w,h),lm_px(lms,LM['l_wrist'],w,h))
        self._rm=max(self._rm,ang); rep=self.rc.update(ang)
        if rep=='rep': self.rc._angle_scores.append(_angle_score(self._rm,150,40)); self._rm=0
        fb='Arms up!' if ang<80 else('Great! 🔥' if ang>120 else 'Keep going!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class RussianTwistAnalyzer:
    label='Russian Twist'
    def __init__(self): self.rc=RepCounter(30,5); self._ls=None; self._t=0; self._tm=0
    def analyze(self,lms,w,h):
        lsh=lm_px(lms,LM['l_shoulder'],w,h);rsh=lm_px(lms,LM['r_shoulder'],w,h)
        lhi=lm_px(lms,LM['l_hip'],w,h);rhi=lm_px(lms,LM['r_hip'],w,h)
        lwr=lm_px(lms,LM['l_wrist'],w,h);rwr=lm_px(lms,LM['r_wrist'],w,h)
        scx=(lsh[0]+rsh[0])/2;hcx=(lhi[0]+rhi[0])/2;rot=abs(scx-hcx)
        wcx=(lwr[0]+rwr[0])/2;side='left' if wcx<hcx else 'right'
        self._tm=max(self._tm,rot)
        if rot>30 and side!=self._ls:
            self._ls=side;self._t+=1
            if self._t%2==0: self.rc.count+=1;self.rc._angle_scores.append(_angle_score(self._tm,80,0));self._tm=0
        self.rc.history.append(rot)
        fb=f'Twist {side}!' if rot<20 else f'Good twist → {side}! 🔥'
        return dict(angle=rot,stage=side,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

ANALYZERS={'Squat':SquatAnalyzer,'Push-Up':PushUpAnalyzer,'Pull-Up':PullUpAnalyzer,
           'Jumping Jack':JumpingJackAnalyzer,'Russian Twist':RussianTwistAnalyzer}

def draw_skeleton(frame,lms,w,h):
    for a,b in SKELETON_EDGES:
        cv2.line(frame,lm_px(lms,a,w,h),lm_px(lms,b,w,h),(57,255,126),2,cv2.LINE_AA)
    for i in range(33):
        pt=lm_px(lms,i,w,h)
        cv2.circle(frame,pt,5,(74,163,22),-1,cv2.LINE_AA)
        cv2.circle(frame,pt,5,(255,255,255),1,cv2.LINE_AA)

def draw_wrong_overlay(frame,suspected):
    H,W=frame.shape[:2];ov=frame.copy()
    cv2.rectangle(ov,(0,0),(W,48),(10,90,200),-1);cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    msg=f"WRONG EXERCISE? Looks like: {suspected.upper()}"
    tw,_=cv2.getTextSize(msg,FONT,0.55,1)[0]
    cv2.putText(frame,msg,(W//2-tw//2,32),FONT,0.55,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,msg,(W//2-tw//2,32),FONT,0.55,(0,200,255),1,cv2.LINE_AA)

def draw_hud(frame,res,ex_title):
    H,W=frame.shape[:2];count=res.get('count',0)
    stage=(res.get('stage') or 'READY').upper();feedback=res.get('feedback','')
    angle_v=float(res.get('angle') or 0)
    label=ex_title.upper();lw,_=cv2.getTextSize(label,FONT,0.65,1)[0]
    cv2.rectangle(frame,(W//2-lw//2-12,6),(W//2+lw//2+12,38),(13,16,20),-1)
    cv2.putText(frame,label,(W//2-lw//2,30),FONT,0.65,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,label,(W//2-lw//2,30),FONT,0.65,(57,255,126),1,cv2.LINE_AA)
    cv2.rectangle(frame,(8,48),(138,178),(13,16,20),-1);cv2.rectangle(frame,(8,48),(138,52),(57,255,126),-1)
    cv2.putText(frame,'REPS',(18,73),FONT,0.42,(100,108,120),1,cv2.LINE_AA)
    cstr=str(count);cw,_=cv2.getTextSize(cstr,FONT,2.6,2)[0]
    cv2.putText(frame,cstr,(73-cw//2,158),FONT,2.6,(0,0,0),5,cv2.LINE_AA)
    cv2.putText(frame,cstr,(73-cw//2,158),FONT,2.6,(57,255,126),2,cv2.LINE_AA)
    sc=(57,255,126) if stage in('UP','READY') else(60,100,235)
    cv2.rectangle(frame,(8,186),(138,212),sc,-1)
    sw,_=cv2.getTextSize(stage,FONT,0.42,1)[0]
    cv2.putText(frame,stage,(73-sw//2,205),FONT,0.42,(0,0,0),2,cv2.LINE_AA)
    cv2.rectangle(frame,(W-148,48),(W-8,178),(13,16,20),-1);cv2.rectangle(frame,(W-148,48),(W-8,52),(235,99,37),-1)
    cv2.putText(frame,'ANGLE',(W-138,73),FONT,0.42,(100,108,120),1,cv2.LINE_AA)
    astr=f'{int(angle_v)} deg';aw,_=cv2.getTextSize(astr,FONT,0.75,1)[0]
    cv2.putText(frame,astr,(W-78-aw//2,130),FONT,0.75,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,astr,(W-78-aw//2,130),FONT,0.75,(235,99,37),1,cv2.LINE_AA)
    if feedback:
        fb=feedback[:58];fw,fh=cv2.getTextSize(fb,FONT,0.5,1)[0];fy2=H-12;fy1=fy2-fh-16
        cv2.rectangle(frame,(W//2-fw//2-18,fy1),(W//2+fw//2+18,fy2),(13,16,20),-1)
        cv2.putText(frame,fb,(W//2-fw//2,fy2-7),FONT,0.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,fb,(W//2-fw//2,fy2-7),FONT,0.5,(255,255,255),1,cv2.LINE_AA)

class EventHistory:
    MAX=100
    def __init__(self): self._lock=threading.Lock(); self._events=deque(maxlen=self.MAX)
    def add(self,kind,message):
        with self._lock:
            self._events.appendleft({'time':datetime.now().strftime('%H:%M:%S'),'kind':kind,'message':message})
    def all(self):
        with self._lock: return list(self._events)
    def clear(self):
        with self._lock: self._events.clear()

_GYM_STATE_VERSION=7

class GymState:
    def __init__(self):
        self._version=_GYM_STATE_VERSION;self.lock=threading.Lock()
        self.result={'count':0,'stage':'','feedback':'Get in position!','angle':0,'form_score':0}
        self.exercise='Squat';self.show_skeleton=True;self.mirror=True
        self.history=EventHistory()
    def set_exercise(self,ex):
        if ex!=self.exercise:
            self.exercise=ex
            self.result={'count':0,'stage':'','feedback':'Get in position!','angle':0,'form_score':0}
            self.history.add('info',f'Switched to {ex}')

def _is_fresh(o): return isinstance(o,GymState) and getattr(o,'_version',0)==_GYM_STATE_VERSION
if not _is_fresh(st.session_state.get('gym_state')): st.session_state.gym_state=GymState()
gym=st.session_state.gym_state

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💪 AI Gym Tracker"); st.markdown("---")
    exercise=st.selectbox("Exercise",list(ANALYZERS.keys()))
    st.markdown("---")
    mode=st.radio("Mode",["📹 Webcam (Real-Time)","📁 Upload Video"],label_visibility="collapsed")
    st.markdown("---")
    show_skeleton=st.checkbox("Show Skeleton",value=True)
    mirror=st.checkbox("Mirror Webcam",value=True)
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🔄 Reset",use_container_width=True):
            gym.result={'count':0,'stage':'','feedback':'Ready!','angle':0,'form_score':0}; st.rerun()
    with c2:
        if st.button("🗑 Clear Log",use_container_width=True): gym.history.clear(); st.rerun()
    st.markdown("---"); st.markdown("### 📋 Exercise Log")
    events=gym.history.all()
    if events:
        bm={'wrong':('WRONG EX','badge-wrong'),'form':('FORM','badge-form'),'info':('INFO','badge-info')}
        p=['<div class="history-log">']
        for ev in events:
            lt,bc=bm.get(ev['kind'],('LOG','badge-info'))
            p.append(f'<div class="log-entry"><span class="log-badge {bc}">{lt}</span>'
                     f'<span class="log-time">{ev["time"]}</span><span class="log-text">{ev["message"]}</span></div>')
        p.append('</div>'); st.markdown(''.join(p),unsafe_allow_html=True)
    else: st.markdown('<div class="no-history">No events yet!</div>',unsafe_allow_html=True)
    st.markdown("---"); st.markdown("**Exercises:**\n🦵 Squat · 💪 Push-Up\n🏋️ Pull-Up · 🙆 Jumping Jack\n🔄 Russian Twist")

gym.set_exercise(exercise); gym.show_skeleton=show_skeleton; gym.mirror=mirror

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("💪 AI GYM TRACKER — REAL-TIME")
st.caption("MediaPipe JS · Runs entirely in your browser · Streamlit Cloud ✅")
col_vid,col_stats=st.columns([3,1])

with col_stats:
    wrong_ex_ph=st.empty(); rep_ph=st.empty(); stage_ph=st.empty()
    angle_ph=st.empty(); fps_ph=st.empty()
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    fb_ph=st.empty()
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    end_btn_ph=st.empty()

def render_stats():
    res=gym.result; cnt=res.get('count',0); s=(res.get('stage') or 'READY').upper()
    ang=int(res.get('angle') or 0); fb=res.get('feedback',''); sc100=res.get('form_score',0)
    sc='#39ff7e' if s in('UP','READY') else '#3b82f6'
    wrong_ex_ph.empty()
    rep_ph.markdown(f'<div class="metric-card"><div class="metric-value">{cnt}</div><div class="metric-label">Reps</div></div>',unsafe_allow_html=True)
    stage_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem;color:{sc};">{s}</div><div class="metric-label">Stage</div></div>',unsafe_allow_html=True)
    angle_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00d4ff;">{ang}°</div><div class="metric-label">Angle</div></div>',unsafe_allow_html=True)
    fps_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.6rem;color:#f59e0b;">JS</div><div class="metric-label">Mode</div></div>',unsafe_allow_html=True)
    if fb: fb_ph.markdown(f'<div class="feedback-box">💬 {fb}</div>',unsafe_allow_html=True)
    _url=f"http://localhost/movera/patient/patient-plan.php?score={sc100}"
    end_btn_ph.markdown(f'<a href="{_url}" target="_blank" style="display:block;width:100%;padding:10px 0;background:#39ff7e;color:#0d0f12;text-align:center;border-radius:10px;font-weight:700;font-size:1rem;text-decoration:none;margin-top:8px;">End Exercise (Score: {sc100})</a>',unsafe_allow_html=True)

render_stats()

# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM MODE — MediaPipe JS runs fully in browser
# ══════════════════════════════════════════════════════════════════════════════
if mode.startswith("📹"):
    with col_vid:
        st.markdown("#### 📹 Live Webcam — Real-Time AI Tracking")

        # JS exercise selector maps to thresholds
        ex_configs = {
            'Squat':        {'joint':[23,25,27],'up':160,'down':90,'higher_is_up':True,  'label':'Knee Angle'},
            'Push-Up':      {'joint':[11,13,15],'up':155,'down':90,'higher_is_up':True,  'label':'Elbow Angle'},
            'Pull-Up':      {'joint':[11,13,15],'up':50, 'down':140,'higher_is_up':False,'label':'Elbow Angle'},
            'Jumping Jack': {'joint':[23,11,15],'up':130,'down':40, 'higher_is_up':True, 'label':'Arm Angle'},
            'Russian Twist':{'joint':[11,13,15],'up':30, 'down':5,  'higher_is_up':True, 'label':'Rotation'},
        }
        cfg = ex_configs[exercise]
        joint_a, joint_b, joint_c = cfg['joint']
        up_thresh   = cfg['up']
        down_thresh = cfg['down']
        higher_is_up = str(cfg['higher_is_up']).lower()
        show_skel    = str(show_skeleton).lower()
        mirror_js    = str(mirror).lower()

        webcam_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;background:#0d0f12;overflow:hidden;}}
  #wrap{{position:relative;width:100%;}}

  /* THE FIX: video must NOT be display:none — use visibility:hidden + absolute
     so it has real dimensions that canvas can read, but isn't visible */
  #vid{{
    position:absolute;top:0;left:0;
    width:640px;height:480px;
    visibility:hidden;
    pointer-events:none;
  }}

  #canvas{{
    width:100%;
    max-width:640px;
    border-radius:10px;
    display:block;
    background:#13161c;
    border:2px solid #1e2229;
  }}
  #placeholder{{
    width:100%;height:360px;border-radius:10px;
    border:2px dashed #2a2f3a;background:#13161c;
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:14px;color:#4b5563;font-size:.9rem;
  }}
  #controls{{display:flex;gap:10px;margin-top:10px;justify-content:center;flex-wrap:wrap;}}
  button{{padding:10px 24px;border-radius:8px;font-size:.9rem;font-weight:700;cursor:pointer;border:none;}}
  #startBtn{{background:#39ff7e;color:#0d0f12;}}
  #stopBtn{{background:#1e2229;color:#e8eaf0;border:1px solid #2a2f3a;display:none;}}
  #statusBar{{margin-top:8px;text-align:center;font-size:.75rem;color:#6b7280;min-height:18px;}}
  .dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
    background:#39ff7e;margin-right:5px;animation:blink 1s infinite;}}
  @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.15}}}}
  #loadingBar{{width:100%;height:3px;background:#1e2229;border-radius:2px;margin-top:6px;overflow:hidden;display:none;}}
  #loadingFill{{height:100%;background:linear-gradient(90deg,#39ff7e,#00d4ff);width:0%;transition:width .3s;}}
</style>
</head>
<body>
<div id="wrap">
  <div id="placeholder">
    <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="1.5">
      <path d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15
               14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
    </svg>
    <span>Click <b style="color:#39ff7e">Start Camera</b></span>
    <span style="font-size:.75rem;color:#374151">MediaPipe runs in your browser</span>
  </div>
  <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
  <video  id="vid" autoplay playsinline muted></video>
</div>
<div id="loadingBar"><div id="loadingFill"></div></div>
<div id="controls">
  <button id="startBtn" onclick="startCam()">&#9654; Start Camera</button>
  <button id="stopBtn"  onclick="stopCam()">&#9632; Stop</button>
</div>
<div id="statusBar">Loading MediaPipe…</div>

<!-- MediaPipe Vision Tasks (CDN) -->
<script type="module">
import {{ PoseLandmarker, FilesetResolver, DrawingUtils }}
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";

// ── Config from Python ────────────────────────────────────────────────────────
const JOINT_A      = {joint_a};
const JOINT_B      = {joint_b};
const JOINT_C      = {joint_c};
const UP_THRESH    = {up_thresh};
const DOWN_THRESH  = {down_thresh};
const HIGHER_IS_UP = {higher_is_up};
const SHOW_SKEL    = {show_skel};
const MIRROR       = {mirror_js};
const EX_LABEL     = "{exercise}";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const video       = document.getElementById('vid');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const placeholder = document.getElementById('placeholder');
const statusEl    = document.getElementById('statusBar');
const loadBar     = document.getElementById('loadingBar');
const loadFill    = document.getElementById('loadingFill');

// ── State ────────────────────────────────────────────────────────────────────
let poseLandmarker = null;
let stream         = null;
let rafHandle      = null;
let running        = false;
let repCount       = 0;
let stage          = null;   // 'up' | 'down'
let lastAngle      = 0;

// ── Colours ───────────────────────────────────────────────────────────────────
const GREEN  = '#39ff7e';
const BLUE   = '#00d4ff';
const ORANGE = '#f59e0b';
const DARK   = 'rgba(13,15,18,0.82)';

// ── Angle helper ──────────────────────────────────────────────────────────────
function anglePx(ax,ay, bx,by, cx,cy) {{
  const ba = [ax-bx, ay-by];
  const bc = [cx-bx, cy-by];
  const dot = ba[0]*bc[0] + ba[1]*bc[1];
  const mag = Math.sqrt(ba[0]**2+ba[1]**2) * Math.sqrt(bc[0]**2+bc[1]**2);
  return Math.acos(Math.max(-1, Math.min(1, dot/(mag+1e-8)))) * 180 / Math.PI;
}}

// ── Rep counter ───────────────────────────────────────────────────────────────
function updateReps(angle) {{
  if (HIGHER_IS_UP) {{
    if (angle > UP_THRESH)                       {{ stage = 'up'; }}
    else if (angle < DOWN_THRESH && stage==='up'){{ stage = 'down'; repCount++; }}
  }} else {{
    if (angle < UP_THRESH)                        {{ stage = 'up'; }}
    else if (angle > DOWN_THRESH && stage==='up') {{ stage = 'down'; repCount++; }}
  }}
}}

// ── Feedback ──────────────────────────────────────────────────────────────────
function getFeedback(angle) {{
  if (EX_LABEL === 'Squat') {{
    if (angle > 110 && stage !== 'down') return 'Go lower!';
    if (angle < 90)  return 'Good depth! 🔥';
    return 'Stand tall!';
  }} else if (EX_LABEL === 'Push-Up') {{
    if (angle > 130 && stage !== 'down') return 'Lower chest!';
    if (angle < 90)  return 'Good depth! 🔥';
    return 'Push up!';
  }} else if (EX_LABEL === 'Pull-Up') {{
    if (angle > 70 && stage === 'up') return 'Pull higher!';
    if (angle < 50)  return 'Chin over bar! 🔥';
    return 'Lower slowly!';
  }} else if (EX_LABEL === 'Jumping Jack') {{
    if (angle < 80)  return 'Arms up!';
    if (angle > 120) return 'Great! 🔥';
    return 'Keep going!';
  }}
  return '';
}}

// ── Draw HUD overlay on canvas ────────────────────────────────────────────────
function drawHUD(angle) {{
  const W = canvas.width, H = canvas.height;
  const stageText = (stage || 'READY').toUpperCase();
  const fb = getFeedback(angle);

  // Exercise label
  ctx.save();
  ctx.font = 'bold 18px sans-serif';
  ctx.textAlign = 'center';
  const lw = ctx.measureText(EX_LABEL).width;
  ctx.fillStyle = DARK;
  ctx.fillRect(W/2-lw/2-12, 6, lw+24, 34);
  ctx.fillStyle = GREEN;
  ctx.fillText(EX_LABEL, W/2, 28);
  ctx.restore();

  // Reps box
  ctx.save();
  ctx.fillStyle = DARK;
  ctx.fillRect(8, 48, 130, 130);
  ctx.fillStyle = GREEN;
  ctx.fillRect(8, 48, 130, 4);
  ctx.font = '11px sans-serif'; ctx.fillStyle = '#6b7280'; ctx.textAlign='center';
  ctx.fillText('REPS', 73, 68);
  ctx.font = 'bold 64px sans-serif'; ctx.fillStyle = GREEN; ctx.textAlign='center';
  ctx.fillText(String(repCount), 73, 155);
  ctx.restore();

  // Stage badge
  ctx.save();
  ctx.fillStyle = (stage==='up') ? GREEN : '#3b6beb';
  ctx.fillRect(8, 186, 130, 28);
  ctx.font = 'bold 12px sans-serif'; ctx.fillStyle = '#000'; ctx.textAlign='center';
  ctx.fillText(stageText, 73, 204);
  ctx.restore();

  // Angle box
  ctx.save();
  ctx.fillStyle = DARK;
  ctx.fillRect(W-140, 48, 132, 130);
  ctx.fillStyle = ORANGE;
  ctx.fillRect(W-140, 48, 132, 4);
  ctx.font = '11px sans-serif'; ctx.fillStyle = '#6b7280'; ctx.textAlign='center';
  ctx.fillText('ANGLE', W-74, 68);
  ctx.font = 'bold 26px sans-serif'; ctx.fillStyle = ORANGE; ctx.textAlign='center';
  ctx.fillText(Math.round(angle) + '°', W-74, 120);
  ctx.restore();

  // Feedback bar
  if (fb) {{
    ctx.save();
    ctx.font = '14px sans-serif';
    const fw = ctx.measureText(fb).width;
    ctx.fillStyle = DARK;
    ctx.fillRect(W/2-fw/2-18, H-36, fw+36, 28);
    ctx.fillStyle = '#fff'; ctx.textAlign='center';
    ctx.fillText(fb, W/2, H-16);
    ctx.restore();
  }}
}}

// ── MediaPipe skeleton colours ────────────────────────────────────────────────
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]
];

function drawSkeleton(landmarks) {{
  const W=canvas.width, H=canvas.height;
  // Connections
  ctx.save();
  ctx.strokeStyle=GREEN; ctx.lineWidth=2;
  for (const [a,b] of POSE_CONNECTIONS) {{
    const la=landmarks[a], lb=landmarks[b];
    ctx.beginPath();
    ctx.moveTo(la.x*W, la.y*H);
    ctx.lineTo(lb.x*W, lb.y*H);
    ctx.stroke();
  }}
  // Joints
  for (const lm of landmarks) {{
    ctx.beginPath();
    ctx.arc(lm.x*W, lm.y*H, 5, 0, 2*Math.PI);
    ctx.fillStyle='#4aa316'; ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.stroke();
  }}
  ctx.restore();
}}

// ── Main render loop ──────────────────────────────────────────────────────────
function renderFrame() {{
  if (!running) return;
  rafHandle = requestAnimationFrame(renderFrame);

  if (video.readyState < 2) return;   // not ready yet

  const W = video.videoWidth  || 640;
  const H = video.videoHeight || 480;

  // Resize canvas to match actual video dimensions
  if (canvas.width !== W)  canvas.width  = W;
  if (canvas.height !== H) canvas.height = H;

  ctx.save();
  if (MIRROR) {{
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
  }}
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

  // Run pose detection
  const result = poseLandmarker.detectForVideo(video, performance.now());

  if (result.landmarks && result.landmarks.length > 0) {{
    const lms = result.landmarks[0];
    if (SHOW_SKEL) drawSkeleton(lms);

    // Compute angle for chosen joints
    const a = lms[JOINT_A], b = lms[JOINT_B], c = lms[JOINT_C];
    const angle = anglePx(
      a.x*W, a.y*H,
      b.x*W, b.y*H,
      c.x*W, c.y*H
    );
    lastAngle = angle;
    updateReps(angle);
  }}

  drawHUD(lastAngle);
}}

// ── Start camera ──────────────────────────────────────────────────────────────
window.startCam = async function() {{
  if (!poseLandmarker) {{
    statusEl.textContent = 'MediaPipe not ready yet — please wait…';
    return;
  }}
  try {{
    statusEl.textContent = 'Requesting camera…';
    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ width: {{ ideal:640 }}, height: {{ ideal:480 }}, facingMode:'user' }},
      audio: false
    }});
    video.srcObject = stream;

    // KEY FIX: wait for loadedmetadata before starting loop
    await new Promise(resolve => {{
      if (video.readyState >= 1) {{ resolve(); return; }}
      video.onloadedmetadata = resolve;
    }});
    await video.play();

    // Now video has real dimensions — show canvas
    placeholder.style.display  = 'none';
    canvas.style.display       = 'block';
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display  = 'inline-block';

    running = true;
    statusEl.innerHTML = '<span class="dot"></span>Live — AI running in browser';
    renderFrame();
  }} catch(e) {{
    statusEl.textContent = 'Camera error: ' + e.message;
  }}
}};

// ── Stop camera ───────────────────────────────────────────────────────────────
window.stopCam = function() {{
  running = false;
  if (rafHandle) cancelAnimationFrame(rafHandle);
  if (stream)    stream.getTracks().forEach(t => t.stop());
  stream = null;
  placeholder.style.display  = 'flex';
  canvas.style.display       = 'none';
  document.getElementById('startBtn').style.display = 'inline-block';
  document.getElementById('stopBtn').style.display  = 'none';
  statusEl.textContent = 'Stopped.';
}};

// ── Load MediaPipe ────────────────────────────────────────────────────────────
(async function loadMediaPipe() {{
  try {{
    statusEl.textContent  = 'Loading MediaPipe WASM…';
    loadBar.style.display = 'block';
    loadFill.style.width  = '30%';

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    loadFill.style.width = '60%';

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {{
      baseOptions: {{
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "GPU"
      }},
      runningMode:          "VIDEO",
      numPoses:             1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    }});

    loadFill.style.width  = '100%';
    setTimeout(() => {{ loadBar.style.display='none'; }}, 400);
    statusEl.textContent  = 'Ready — click Start Camera';
    document.getElementById('startBtn').disabled = false;
  }} catch(e) {{
    statusEl.textContent = 'Failed to load MediaPipe: ' + e.message;
    console.error(e);
  }}
}})();
</script>
</body>
</html>"""

        components.html(webcam_html, height=580, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD MODE — Python MediaPipe (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
else:
    with col_vid:
        st.markdown("#### 📁 Upload Video for Analysis")
        uploaded=st.file_uploader("MP4 / AVI / MOV",type=["mp4","avi","mov","mkv"])
        if uploaded:
            st.video(uploaded)
            if st.button("🧠 Analyze Video",type="primary",use_container_width=True):
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python.vision import PoseLandmarker,PoseLandmarkerOptions
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
                with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as f:
                    f.write(uploaded.read()); tmp_path=f.name
                opts=PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionTaskRunningMode.VIDEO,num_poses=1,
                    min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5)
                lm=PoseLandmarker.create_from_options(opts)
                cap=cv2.VideoCapture(tmp_path)
                vw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.; tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                analyzer=ANALYZERS[exercise]()
                prog=st.progress(0,"Processing…"); preview=st.empty()
                last_res={'count':0,'stage':'','feedback':'','angle':0}; fidx=0
                gym.history.add('info',f'Video started: {exercise}')
                while True:
                    ret,frame=cap.read()
                    if not ret: break
                    fidx+=1
                    det=lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),int(fidx/fps*1000))
                    if det.pose_landmarks:
                        lms=det.pose_landmarks[0]
                        if show_skeleton: draw_skeleton(frame,lms,vw,vh)
                        last_res=analyzer.analyze(lms,vw,vh)
                    else: last_res={**last_res,'feedback':'No pose'}
                    draw_hud(frame,last_res,exercise)
                    if fidx%30==0 or fidx==tot:
                        preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True)
                        prog.progress(min(fidx/max(tot,1),1.),text=f"Frame {fidx}/{tot} — Reps:{last_res['count']}")
                cap.release(); lm.close(); os.unlink(tmp_path); prog.empty()
                gym.history.add('info',f'Done — {last_res["count"]} reps')
                st.success(f"✅ {last_res['count']} reps in {fidx} frames.")
                gym.result=last_res; render_stats()
                hist=list(analyzer.rc.history)
                if hist:
                    import pandas as pd
                    st.markdown("#### 📈 Angle History")
                    st.line_chart(pd.DataFrame({'angle':hist}),color="#39ff7e")
