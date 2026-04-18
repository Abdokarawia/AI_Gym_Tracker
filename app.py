"""
AI Gym Tracker — Streamlit Cloud Compatible (Proper Custom Component)
======================================================================
No WebRTC · No STUN/TURN · Works on Streamlit Cloud free tier

Architecture:
  1. webcam_component/index.html  — proper declare_component() frontend
     • Captures webcam via getUserMedia
     • Sends base64 JPEG → Python via Streamlit.setComponentValue()
     • Receives annotated frame back via component args["annotated"]
     • Renders annotated frame on <canvas>

  2. Python (this file)
     • Decodes b64 → numpy → MediaPipe → draw HUD → encode → b64
     • Passes annotated frame back to component on next rerun

  NO st.components.v1.html  (deprecated — Streamlit.setComponentValue unavailable there)
  NO streamlit-webrtc        (requires TURN/STUN)
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
from pathlib import Path
from collections import deque
from datetime import datetime

st.set_page_config(
    page_title="AI Gym Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Register custom webcam component ─────────────────────────────────────────
# declare_component serves index.html from the local path and injects the
# Streamlit JS bridge — window.Streamlit is available, setComponentValue works.
_COMPONENT_DIR = Path(__file__).parent / "webcam_component"
_webcam_component = components.declare_component(
    "webcam_component",
    path=str(_COMPONENT_DIR),
)

def webcam_component(annotated: str = "", key: str = "webcam"):
    """
    Render the webcam component.
    annotated : base64 JPEG data-URI to display (Python → JS)
    Returns   : base64 JPEG from browser webcam (JS → Python), or None
    """
    return _webcam_component(annotated=annotated, key=key, default=None)


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
.history-log{max-height:320px;overflow-y:auto;padding-right:4px;}
.log-entry{display:flex;gap:10px;align-items:flex-start;padding:8px 10px;margin-bottom:5px;border-radius:8px;background:#13161c;border:1px solid #1e2229;font-size:.82rem;line-height:1.4;}
.log-badge{flex-shrink:0;padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-top:1px;}
.badge-wrong{background:#78350f;color:#fcd34d;}
.badge-form{background:#1e1b4b;color:#a5b4fc;}
.badge-info{background:#052e16;color:#6ee7b7;}
.log-time{color:#4b5563;font-size:.72rem;flex-shrink:0;margin-top:2px;}
.log-text{color:#d1d5db;flex:1;}
.no-history{color:#374151;font-size:.82rem;font-style:italic;text-align:center;padding:20px 0;}
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

# ── Pose geometry ─────────────────────────────────────────────────────────────
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

# ── Rep counter ───────────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self,up,down,higher_is_up=True):
        self.up_thresh=up;self.down_thresh=down;self.higher_is_up=higher_is_up
        self.count=0;self.stage=None;self.history=deque(maxlen=2000)
        self._rep_best=None;self._angle_scores=[]
    def update(self,v):
        self.history.append(v)
        if self._rep_best is None:self._rep_best=v
        if self.higher_is_up:
            self._rep_best=min(self._rep_best,v)
            if v>self.up_thresh:self.stage='up';self._rep_best=None
            elif v<self.down_thresh and self.stage=='up':self.stage='down';self.count+=1;return'rep'
        else:
            self._rep_best=max(self._rep_best,v)
            if v<self.up_thresh:self.stage='up';self._rep_best=None
            elif v>self.down_thresh and self.stage=='up':self.stage='down';self.count+=1;return'rep'
        return''
    def angle_score(self):return int(sum(self._angle_scores)/len(self._angle_scores)) if self._angle_scores else 0
    def reset(self):self.count=0;self.stage=None;self.history.clear();self._rep_best=None;self._angle_scores=[]

# ── Wrong-exercise detector ───────────────────────────────────────────────────
class WrongExerciseDetector:
    WARN_FRAMES=20;COOLDOWN=90
    def __init__(self):self._mm=0;self._lw=0;self._fi=0
    def _scores(self,lms,w,h):
        s={}
        def ang(a,b,c):return angle3(lm_px(lms,a,w,h),lm_px(lms,b,w,h),lm_px(lms,c,w,h))
        ka=ang(LM['l_hip'],LM['l_knee'],LM['l_ankle'])
        hh=lm_px(lms,LM['l_hip'],w,h)[1]/h
        s['Squat']=min(1.,max(0,1-abs(ka-90)/90)*(0.4+0.6*hh))
        ea=ang(LM['l_shoulder'],LM['l_elbow'],LM['l_wrist'])
        shy=lm_px(lms,LM['l_shoulder'],w,h)[1]/h;hy=lm_px(lms,LM['l_hip'],w,h)[1]/h
        s['Push-Up']=min(1.,max(0,1-abs(ea-90)/90)*(0.3+0.7*(1-abs(shy-hy))))
        wry=lm_px(lms,LM['l_wrist'],w,h)[1]/h;sh2=lm_px(lms,LM['l_shoulder'],w,h)[1]/h
        s['Pull-Up']=min(1.,max(0,1-abs(ea-60)/80)*(0.3+0.7*max(0,sh2-wry)*4))
        aa=ang(LM['l_hip'],LM['l_shoulder'],LM['l_wrist'])
        s['Jumping Jack']=min(1.,max(0,(aa-60)/100))
        lx=lm_px(lms,LM['l_wrist'],w,h)[0]/w;rx=lm_px(lms,LM['r_wrist'],w,h)[0]/w
        s['Russian Twist']=min(1.,max(0,abs(lx-rx)-0.05)*2)
        return s
    def check(self,sel,lms,w,h):
        self._fi+=1;sc=self._scores(lms,w,h)
        bx=max(sc,key=sc.__getitem__);bs=sc[bx];ss=sc.get(sel,0)
        mm=(bx!=sel and bs-ss>0.30 and bs>0.35)
        self._mm=self._mm+1 if mm else max(0,self._mm-2)
        ok=(self._fi-self._lw)>self.COOLDOWN
        if self._mm>=self.WARN_FRAMES and ok:
            self._lw=self._fi;self._mm=0;return True,bx,round(bs,2)
        return False,bx,round(bs,2)

# ── Exercise analyzers ────────────────────────────────────────────────────────
class SquatAnalyzer:
    label='Squat'
    def __init__(self):self.rc=RepCounter(160,90)
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_hip'],w,h),lm_px(lms,LM['l_knee'],w,h),lm_px(lms,LM['l_ankle'],w,h))
        rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None:self.rc._angle_scores.append(_angle_score(self.rc._rep_best,70,160))
        fb='Go lower!' if ang>110 and self.rc.stage!='down' else('Good depth! 🔥' if ang<90 else 'Stand tall!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class PushUpAnalyzer:
    label='Push-Up'
    def __init__(self):self.rc=RepCounter(155,90)
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_shoulder'],w,h),lm_px(lms,LM['l_elbow'],w,h),lm_px(lms,LM['l_wrist'],w,h))
        rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None:self.rc._angle_scores.append(_angle_score(self.rc._rep_best,60,155))
        fb='Lower chest!' if ang>130 and self.rc.stage!='down' else('Good depth! 🔥' if ang<90 else 'Push up!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class PullUpAnalyzer:
    label='Pull-Up'
    def __init__(self):self.rc=RepCounter(50,140,higher_is_up=False)
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_shoulder'],w,h),lm_px(lms,LM['l_elbow'],w,h),lm_px(lms,LM['l_wrist'],w,h))
        rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None:self.rc._angle_scores.append(_angle_score(self.rc._rep_best,30,140))
        fb='Pull higher!' if ang>70 and self.rc.stage=='up' else('Chin over bar! 🔥' if ang<50 else 'Lower slowly!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class JumpingJackAnalyzer:
    label='Jumping Jack'
    def __init__(self):self.rc=RepCounter(130,40);self._rm=0
    def analyze(self,lms,w,h):
        ang=angle3(lm_px(lms,LM['l_hip'],w,h),lm_px(lms,LM['l_shoulder'],w,h),lm_px(lms,LM['l_wrist'],w,h))
        self._rm=max(self._rm,ang);rep=self.rc.update(ang)
        if rep=='rep':self.rc._angle_scores.append(_angle_score(self._rm,150,40));self._rm=0
        fb='Arms up!' if ang<80 else('Great! 🔥' if ang>120 else 'Keep going!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class RussianTwistAnalyzer:
    label='Russian Twist'
    def __init__(self):self.rc=RepCounter(30,5);self._ls=None;self._t=0;self._tm=0
    def analyze(self,lms,w,h):
        lsh=lm_px(lms,LM['l_shoulder'],w,h);rsh=lm_px(lms,LM['r_shoulder'],w,h)
        lhi=lm_px(lms,LM['l_hip'],w,h);rhi=lm_px(lms,LM['r_hip'],w,h)
        lwr=lm_px(lms,LM['l_wrist'],w,h);rwr=lm_px(lms,LM['r_wrist'],w,h)
        scx=(lsh[0]+rsh[0])/2;hcx=(lhi[0]+rhi[0])/2;rot=abs(scx-hcx)
        wcx=(lwr[0]+rwr[0])/2;side='left' if wcx<hcx else 'right'
        self._tm=max(self._tm,rot)
        if rot>30 and side!=self._ls:
            self._ls=side;self._t+=1
            if self._t%2==0:self.rc.count+=1;self.rc._angle_scores.append(_angle_score(self._tm,80,0));self._tm=0
        self.rc.history.append(rot)
        fb=f'Twist {side}!' if rot<20 else f'Good twist → {side}! 🔥'
        return dict(angle=rot,stage=side,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

ANALYZERS={'Squat':SquatAnalyzer,'Push-Up':PushUpAnalyzer,'Pull-Up':PullUpAnalyzer,
           'Jumping Jack':JumpingJackAnalyzer,'Russian Twist':RussianTwistAnalyzer}

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_skeleton(frame,lms,w,h):
    for a,b in SKELETON_EDGES:cv2.line(frame,lm_px(lms,a,w,h),lm_px(lms,b,w,h),(57,255,126),2,cv2.LINE_AA)
    for i in range(33):
        pt=lm_px(lms,i,w,h);cv2.circle(frame,pt,5,(74,163,22),-1,cv2.LINE_AA);cv2.circle(frame,pt,5,(255,255,255),1,cv2.LINE_AA)

def draw_wrong_ex_overlay(frame,suspected):
    H,W=frame.shape[:2];ov=frame.copy()
    cv2.rectangle(ov,(0,0),(W,48),(10,90,200),-1);cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    msg=f"WRONG EXERCISE? Looks like: {suspected.upper()}";tw,_=cv2.getTextSize(msg,FONT,0.55,1)[0]
    cv2.putText(frame,msg,(W//2-tw//2,32),FONT,0.55,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,msg,(W//2-tw//2,32),FONT,0.55,(0,200,255),1,cv2.LINE_AA)

def draw_hud(frame,res,ex_title):
    H,W=frame.shape[:2];count=res.get('count',0)
    stage=(res.get('stage') or 'READY').upper();feedback=res.get('feedback','');angle_v=float(res.get('angle') or 0)
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
    cv2.rectangle(frame,(8,186),(138,212),sc,-1);sw,_=cv2.getTextSize(stage,FONT,0.42,1)[0]
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

# ── Event history ─────────────────────────────────────────────────────────────
class EventHistory:
    MAX=100
    def __init__(self):self._lock=threading.Lock();self._events=deque(maxlen=self.MAX)
    def add(self,kind,message):
        with self._lock:self._events.appendleft({'time':datetime.now().strftime('%H:%M:%S'),'kind':kind,'message':message})
    def all(self):
        with self._lock:return list(self._events)
    def clear(self):
        with self._lock:self._events.clear()

# ── Gym state ─────────────────────────────────────────────────────────────────
_GYM_STATE_VERSION=4

class GymState:
    def __init__(self):
        self._version=_GYM_STATE_VERSION;self.lock=threading.Lock()
        self.result={'count':0,'stage':'','feedback':'Get in position!','angle':0,'form_score':0}
        self.exercise='Squat';self.show_skeleton=True;self.mirror=True
        self.analyzer=SquatAnalyzer();self.wrong_ex_det=WrongExerciseDetector()
        self.wrong_ex_flag=False;self.wrong_ex_name=''
        self._mp=None;self._landmarker=None;self.history=EventHistory()
        self._last_feedback='';self._fps_frames=0;self._fps_last=time.time();self.current_fps=0.

    def set_exercise(self,ex):
        with self.lock:
            if ex!=self.exercise:
                self.exercise=ex;self.analyzer=ANALYZERS[ex]()
                self.wrong_ex_det=WrongExerciseDetector();self.wrong_ex_flag=False;self.wrong_ex_name=''
                self.result={'count':0,'stage':'','feedback':'Get in position!','angle':0,'form_score':0}
                self._close_landmarker();self.history.add('info',f'Switched to {ex}')

    def _close_landmarker(self):
        if self._landmarker:
            try:self._landmarker.close()
            except:pass
            self._landmarker=None

    def get_landmarker(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import PoseLandmarker,PoseLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            self._mp=mp
            self._landmarker=PoseLandmarker.create_from_options(PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionTaskRunningMode.IMAGE,num_poses=1,
                min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5))
        return self._mp,self._landmarker

    def process_b64_frame(self,b64_data:str)->str:
        try:
            _,enc=b64_data.split(',',1) if ','in b64_data else('',b64_data)
            frame=cv2.imdecode(np.frombuffer(base64.b64decode(enc),np.uint8),cv2.IMREAD_COLOR)
            if frame is None:return''
        except:return''
        with self.lock:
            H,W=frame.shape[:2]
            if self.mirror:frame=cv2.flip(frame,1)
            mp,lm=self.get_landmarker()
            det=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            if det.pose_landmarks:
                lms=det.pose_landmarks[0]
                if self.show_skeleton:draw_skeleton(frame,lms,W,H)
                is_w,sus,conf=self.wrong_ex_det.check(self.exercise,lms,W,H)
                if is_w:
                    self.wrong_ex_flag=True;self.wrong_ex_name=sus
                    self.history.add('wrong',f'Detected {sus} (conf {conf:.0%}) — selected:{self.exercise}')
                    draw_wrong_ex_overlay(frame,sus)
                else:self.wrong_ex_flag=False
                self.result=self.analyzer.analyze(lms,W,H)
                fb=self.result.get('feedback','')
                if fb and fb!=self._last_feedback:
                    self._last_feedback=fb
                    if any(k in fb.lower() for k in['lower','higher','go','pull','push','twist','arms','chin','chest']):
                        self.history.add('form',fb)
            else:self.result={**self.result,'feedback':'No pose — step back & stand tall'}
            draw_hud(frame,self.result,self.exercise)
            self._fps_frames+=1;now=time.time()
            if now-self._fps_last>=2.:
                self.current_fps=self._fps_frames/(now-self._fps_last);self._fps_frames=0;self._fps_last=now
            _,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            return'data:image/jpeg;base64,'+base64.b64encode(buf).decode()

def _is_fresh(o):return isinstance(o,GymState) and getattr(o,'_version',0)==_GYM_STATE_VERSION
if not _is_fresh(st.session_state.get('gym_state')):st.session_state.gym_state=GymState()
gym=st.session_state.gym_state

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💪 AI Gym Tracker");st.markdown("---")
    exercise=st.selectbox("Exercise",list(ANALYZERS.keys()))
    st.markdown("---")
    mode=st.radio("Mode",["📹 Webcam (Real-Time)","📁 Upload Video"],label_visibility="collapsed")
    st.markdown("---")
    show_skeleton=st.checkbox("Show Skeleton",value=True);mirror=st.checkbox("Mirror Webcam",value=True)
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🔄 Reset",use_container_width=True):
            with gym.lock:gym.analyzer.rc.reset();gym.result={'count':0,'stage':'','feedback':'Ready!','angle':0,'form_score':0}
            st.rerun()
    with c2:
        if st.button("🗑 Clear Log",use_container_width=True):gym.history.clear();st.rerun()
    st.markdown("---");st.markdown("### 📋 Exercise Log")
    events=gym.history.all()
    if events:
        bm={'wrong':('WRONG EX','badge-wrong'),'form':('FORM','badge-form'),'info':('INFO','badge-info')}
        p=['<div class="history-log">']
        for ev in events:
            lt,bc=bm.get(ev['kind'],('LOG','badge-info'))
            p.append(f'<div class="log-entry"><span class="log-badge {bc}">{lt}</span>'
                     f'<span class="log-time">{ev["time"]}</span><span class="log-text">{ev["message"]}</span></div>')
        p.append('</div>');st.markdown(''.join(p),unsafe_allow_html=True)
    else:st.markdown('<div class="no-history">No events yet — start exercising!</div>',unsafe_allow_html=True)
    st.markdown("---");st.markdown("**Exercises:**\n🦵 Squat · 💪 Push-Up\n🏋️ Pull-Up · 🙆 Jumping Jack\n🔄 Russian Twist")

gym.set_exercise(exercise);gym.show_skeleton=show_skeleton;gym.mirror=mirror

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("💪 AI GYM TRACKER — REAL-TIME")
st.caption("MediaPipe Pose · Custom Component · No WebRTC · Works on Streamlit Cloud free tier")
col_vid,col_stats=st.columns([3,1])

with col_stats:
    wrong_ex_ph=st.empty();rep_ph=st.empty();stage_ph=st.empty()
    angle_ph=st.empty();fps_ph=st.empty()
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    fb_ph=st.empty()
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    end_btn_ph=st.empty()

def render_stats():
    res=gym.result;cnt=res.get('count',0);s=(res.get('stage') or 'READY').upper()
    ang=int(res.get('angle') or 0);fb=res.get('feedback','');sc100=res.get('form_score',0)
    sc='#39ff7e' if s in('UP','READY') else'#3b82f6'
    if gym.wrong_ex_flag:
        wrong_ex_ph.markdown(f'<div class="wrong-ex-banner">⚠️ Wrong exercise detected!<br>'
            f'<strong>Looks like: {gym.wrong_ex_name}</strong><br>'
            f'<small>Switch selector or adjust position</small></div>',unsafe_allow_html=True)
    else:wrong_ex_ph.empty()
    rep_ph.markdown(f'<div class="metric-card"><div class="metric-value">{cnt}</div><div class="metric-label">Reps</div></div>',unsafe_allow_html=True)
    stage_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem;color:{sc};">{s}</div><div class="metric-label">Stage</div></div>',unsafe_allow_html=True)
    angle_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00d4ff;">{ang}°</div><div class="metric-label">Angle</div></div>',unsafe_allow_html=True)
    fps_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.6rem;color:#f59e0b;">{gym.current_fps:.1f}</div><div class="metric-label">FPS</div></div>',unsafe_allow_html=True)
    if fb:fb_ph.markdown(f'<div class="feedback-box">💬 {fb}</div>',unsafe_allow_html=True)
    _url=f"http://localhost/movera/patient/patient-plan.php?score={sc100}"
    end_btn_ph.markdown(f'<a href="{_url}" target="_blank" rel="noopener noreferrer" style="display:block;width:100%;padding:10px 0;background:#39ff7e;color:#0d0f12;text-align:center;border-radius:10px;font-weight:700;font-size:1rem;text-decoration:none;margin-top:8px;">End Exercise (Score: {sc100})</a>',unsafe_allow_html=True)

render_stats()

# ── Webcam mode ───────────────────────────────────────────────────────────────
if mode.startswith("📹"):
    with col_vid:
        st.markdown("#### 📹 Live Webcam — Real-Time AI Tracking")
        annotated_b64=st.session_state.get('annotated_frame','')
        # Render component: sends raw frame JS→Python, receives annotated Python→JS
        raw_frame=webcam_component(annotated=annotated_b64,key=f"webcam_{exercise}")
        if raw_frame and isinstance(raw_frame,str) and len(raw_frame)>200:
            annotated=gym.process_b64_frame(raw_frame)
            if annotated:
                st.session_state['annotated_frame']=annotated
                render_stats();st.rerun()

# ── Upload mode ───────────────────────────────────────────────────────────────
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
                    f.write(uploaded.read());tmp_path=f.name
                opts=PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionTaskRunningMode.VIDEO,num_poses=1,
                    min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5,min_tracking_confidence=0.5)
                lm=PoseLandmarker.create_from_options(opts)
                cap=cv2.VideoCapture(tmp_path)
                vw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));vh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.;tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                analyzer=ANALYZERS[exercise]();det_video=WrongExerciseDetector()
                prog=st.progress(0,"Processing…");preview=st.empty()
                last_res={'count':0,'stage':'','feedback':'','angle':0};fidx=0
                gym.history.add('info',f'Video analysis started: {exercise}')
                while True:
                    ret,frame=cap.read()
                    if not ret:break
                    fidx+=1
                    det=lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),int(fidx/fps*1000))
                    if det.pose_landmarks:
                        lms=det.pose_landmarks[0]
                        if show_skeleton:draw_skeleton(frame,lms,vw,vh)
                        is_w,sus,conf=det_video.check(exercise,lms,vw,vh)
                        if is_w:draw_wrong_ex_overlay(frame,sus);gym.history.add('wrong',f'[Frame {fidx}] {sus} ({conf:.0%})')
                        last_res=analyzer.analyze(lms,vw,vh)
                    else:last_res={**last_res,'feedback':'No pose detected'}
                    draw_hud(frame,last_res,exercise)
                    if fidx%30==0 or fidx==tot:
                        preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True)
                        prog.progress(min(fidx/max(tot,1),1.),text=f"Frame {fidx}/{tot} — Reps:{last_res['count']}")
                cap.release();lm.close();os.unlink(tmp_path);prog.empty()
                gym.history.add('info',f'Video done — {last_res["count"]} reps')
                st.success(f"✅ Done! **{last_res['count']} reps** in {fidx} frames.")
                gym.result=last_res;render_stats()
                hist=list(analyzer.rc.history)
                if hist:
                    import pandas as pd
                    st.markdown("#### 📈 Angle History")
                    st.line_chart(pd.DataFrame({'angle':hist}),color="#39ff7e")
