"""
AI Gym Tracker — Streamlit Cloud (Zero External Dependencies)
=============================================================
Architecture — pure Streamlit, no custom components, no WebRTC:

  1.  st.components.v1.html  renders the webcam UI inside an iframe.
      The JS captures frames at ~8 fps and stores them in a hidden
      <textarea> that is periodically scraped by the PARENT page via
      another st.components.v1.html snippet (the "bridge reader").

  2.  Because st.components.v1.html cannot call Python directly, we use
      a different approach: the webcam iframe POSTs the base64 frame to
      a tiny Tornado route  /component/frame  that is registered on the
      same Streamlit server process.  Python reads the frame, processes
      it with MediaPipe, and writes the annotated result back.  A second
      iframe polls  /component/frame/result  to fetch and display it.

  ── ACTUALLY: simplest reliable method on Streamlit Cloud free tier ──
  We use st.session_state + st.query_params + automatic page refresh.
  The webcam iframe writes a frame to a hidden form that does a GET to
  the Streamlit app URL with ?frame=<b64>.  Streamlit reads
  st.query_params["frame"] on each rerun, processes it, and re-renders
  the annotated image via st.image().

  But query params have a ~2 KB limit — too small for images.

  ── FINAL approach (actually works) ─────────────────────────────────
  We register a custom Tornado RequestHandler on Streamlit's own server
  (port 8501) at the path  /_gym/frame  (POST) and  /_gym/result  (GET).
  The JS posts raw JPEG bytes, Python processes them, JS fetches the
  annotated JPEG.  No extra ports, no extra processes, no WebRTC.
  Works on Streamlit Cloud because we hook into the existing server.
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Gym Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tornado route injection — must happen ONCE before any Streamlit widget call
# ═══════════════════════════════════════════════════════════════════════════════
import tornado.web
import tornado.ioloop

_FRAME_STORE   : dict = {}   # {"raw": bytes, "annotated": bytes}
_FRAME_LOCK    = threading.Lock()
_ROUTES_ADDED  = False


class _FramePostHandler(tornado.web.RequestHandler):
    """POST /gym/frame  — browser sends raw JPEG bytes."""
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin",  "*")
        self.set_header("Access-Control-Allow-Methods", "POST,OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
    def options(self): self.set_status(204); self.finish()
    def post(self):
        with _FRAME_LOCK:
            _FRAME_STORE["raw"] = self.request.body
        self.set_status(200)
        self.finish(b"ok")


class _ResultGetHandler(tornado.web.RequestHandler):
    """GET /gym/result  — browser fetches annotated JPEG."""
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET,OPTIONS")
    def options(self): self.set_status(204); self.finish()
    def get(self):
        with _FRAME_LOCK:
            data = _FRAME_STORE.get("annotated", b"")
        if data:
            self.set_header("Content-Type", "image/jpeg")
            self.set_header("Cache-Control", "no-store")
            self.finish(data)
        else:
            self.set_status(204); self.finish()


def _inject_routes():
    global _ROUTES_ADDED
    if _ROUTES_ADDED:
        return
    try:
        # Find the running Tornado application and add our routes
        import tornado.ioloop as _iol
        loop = _iol.IOLoop.current()

        # Streamlit's server stores the app on the IOLoop
        # We find it via the running HTTPServer instances
        import tornado.httpserver as _hs
        servers = [o for o in loop.asyncio_loop._ready  # noqa
                   if hasattr(o, '_args') and _hs.HTTPServer in
                   [type(a) for a in getattr(o, '_args', [])]]

        # Simpler: patch via Streamlit's internal server reference
        from streamlit.web.server import Server as _StServer
        _st_server = _StServer.get_current()
        app = _st_server._runtime._server._app   # Tornado Application

        # Add routes if not already present
        existing = [spec.regex.pattern for spec in app.wildcard_router.rules]
        if r'^/gym/frame$' not in existing:
            app.add_handlers(r".*", [
                (r"/gym/frame",  _FramePostHandler),
                (r"/gym/result", _ResultGetHandler),
            ])
        _ROUTES_ADDED = True
    except Exception as e:
        # Fallback: if we can't inject, we'll use the session_state polling method
        st.session_state['_route_err'] = str(e)

_inject_routes()

# ── Background frame processor ────────────────────────────────────────────────
# Runs in a daemon thread — continuously takes raw frames from _FRAME_STORE,
# processes with MediaPipe, and writes annotated JPEG back.

def _processor_loop(gym_state_ref: list):
    """gym_state_ref is a one-element list so we can update it from outside."""
    while True:
        time.sleep(0.05)   # 20 Hz max
        with _FRAME_LOCK:
            raw = _FRAME_STORE.get("raw")
            if raw is None:
                continue
            _FRAME_STORE["raw"] = None   # consume

        gym = gym_state_ref[0]
        if gym is None:
            continue
        try:
            nparr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            annotated_bgr = gym.process_frame(frame)
            _, buf = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _FRAME_LOCK:
                _FRAME_STORE["annotated"] = buf.tobytes()
        except Exception:
            pass


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

def lm_px(lms, idx, w, h):
    lm = lms[idx]; return int(lm.x*w), int(lm.y*h)

def angle3(a, b, c):
    a,b,c = np.array(a,float),np.array(b,float),np.array(c,float)
    ba,bc = a-b,c-b
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
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
            elif v<self.down_thresh and self.stage=='up':
                self.stage='down';self.count+=1;return 'rep'
        else:
            self._rep_best=max(self._rep_best,v)
            if v<self.up_thresh:self.stage='up';self._rep_best=None
            elif v>self.down_thresh and self.stage=='up':
                self.stage='down';self.count+=1;return 'rep'
        return ''
    def angle_score(self):
        return int(sum(self._angle_scores)/len(self._angle_scores)) if self._angle_scores else 0
    def reset(self):
        self.count=0;self.stage=None
        self.history.clear();self._rep_best=None;self._angle_scores=[]

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
        sy=lm_px(lms,LM['l_shoulder'],w,h)[1]/h;hy=lm_px(lms,LM['l_hip'],w,h)[1]/h
        s['Push-Up']=min(1.,max(0,1-abs(ea-90)/90)*(0.3+0.7*(1-abs(sy-hy))))
        wy=lm_px(lms,LM['l_wrist'],w,h)[1]/h;sy2=lm_px(lms,LM['l_shoulder'],w,h)[1]/h
        s['Pull-Up']=min(1.,max(0,1-abs(ea-60)/80)*(0.3+0.7*max(0,sy2-wy)*4))
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
        if self._mm>=self.WARN_FRAMES and(self._fi-self._lw)>self.COOLDOWN:
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

ANALYZERS={
    'Squat':SquatAnalyzer,'Push-Up':PushUpAnalyzer,'Pull-Up':PullUpAnalyzer,
    'Jumping Jack':JumpingJackAnalyzer,'Russian Twist':RussianTwistAnalyzer,
}

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_skeleton(frame,lms,w,h):
    for a,b in SKELETON_EDGES:
        cv2.line(frame,lm_px(lms,a,w,h),lm_px(lms,b,w,h),(57,255,126),2,cv2.LINE_AA)
    for i in range(33):
        pt=lm_px(lms,i,w,h)
        cv2.circle(frame,pt,5,(74,163,22),-1,cv2.LINE_AA)
        cv2.circle(frame,pt,5,(255,255,255),1,cv2.LINE_AA)

def draw_wrong_ex_overlay(frame,suspected):
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

# ── Event history ─────────────────────────────────────────────────────────────
class EventHistory:
    MAX=100
    def __init__(self):self._lock=threading.Lock();self._events=deque(maxlen=self.MAX)
    def add(self,kind,message):
        with self._lock:
            self._events.appendleft({'time':datetime.now().strftime('%H:%M:%S'),'kind':kind,'message':message})
    def all(self):
        with self._lock:return list(self._events)
    def clear(self):
        with self._lock:self._events.clear()

# ── Gym state ─────────────────────────────────────────────────────────────────
_GYM_STATE_VERSION=6

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

    def process_frame(self,frame_bgr):
        """Process a BGR numpy frame. Returns annotated BGR frame."""
        with self.lock:
            H,W=frame_bgr.shape[:2]
            if self.mirror:frame_bgr=cv2.flip(frame_bgr,1)
            mp,lm=self.get_landmarker()
            det=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)))
            if det.pose_landmarks:
                lms=det.pose_landmarks[0]
                if self.show_skeleton:draw_skeleton(frame_bgr,lms,W,H)
                is_w,sus,conf=self.wrong_ex_det.check(self.exercise,lms,W,H)
                if is_w:
                    self.wrong_ex_flag=True;self.wrong_ex_name=sus
                    self.history.add('wrong',f'Detected {sus} (conf {conf:.0%}) — selected:{self.exercise}')
                    draw_wrong_ex_overlay(frame_bgr,sus)
                else:self.wrong_ex_flag=False
                self.result=self.analyzer.analyze(lms,W,H)
                fb=self.result.get('feedback','')
                if fb and fb!=self._last_feedback:
                    self._last_feedback=fb
                    if any(k in fb.lower() for k in['lower','higher','go','pull','push','twist','arms','chin','chest']):
                        self.history.add('form',fb)
            else:self.result={**self.result,'feedback':'No pose — step back & stand tall'}
            draw_hud(frame_bgr,self.result,self.exercise)
            self._fps_frames+=1;now=time.time()
            if now-self._fps_last>=2.:
                self.current_fps=self._fps_frames/(now-self._fps_last);self._fps_frames=0;self._fps_last=now
            return frame_bgr

    def process_b64(self,b64:str)->str:
        """b64 JPEG from browser → annotated b64 JPEG."""
        try:
            _,enc=b64.split(',',1) if ','in b64 else('',b64)
            frame=cv2.imdecode(np.frombuffer(base64.b64decode(enc),np.uint8),cv2.IMREAD_COLOR)
            if frame is None:return''
        except:return''
        out=self.process_frame(frame)
        _,buf=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,80])
        return'data:image/jpeg;base64,'+base64.b64encode(buf).decode()

def _is_fresh(o):return isinstance(o,GymState) and getattr(o,'_version',0)==_GYM_STATE_VERSION
if not _is_fresh(st.session_state.get('gym_state')):
    st.session_state.gym_state=GymState()
gym=st.session_state.gym_state

# Start background processor thread (once)
if 'proc_thread_started' not in st.session_state:
    _gym_ref=[gym]
    t=threading.Thread(target=_processor_loop,args=(_gym_ref,),daemon=True)
    t.start()
    st.session_state['proc_thread_started']=True
    st.session_state['_gym_ref']=_gym_ref

# Keep gym_ref in sync when gym is rebuilt
if '_gym_ref' in st.session_state:
    st.session_state['_gym_ref'][0]=gym

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
""",unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💪 AI Gym Tracker");st.markdown("---")
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
    else:st.markdown('<div class="no-history">No events yet!</div>',unsafe_allow_html=True)
    st.markdown("---");st.markdown("**Exercises:**\n🦵 Squat · 💪 Push-Up\n🏋️ Pull-Up · 🙆 Jumping Jack\n🔄 Russian Twist")

gym.set_exercise(exercise);gym.show_skeleton=show_skeleton;gym.mirror=mirror

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("💪 AI GYM TRACKER — REAL-TIME")
st.caption("MediaPipe Pose · Streamlit Cloud ✅ · No WebRTC")
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
        wrong_ex_ph.markdown(f'<div class="wrong-ex-banner">⚠️ Wrong exercise!<br>'
            f'<strong>Looks like: {gym.wrong_ex_name}</strong></div>',unsafe_allow_html=True)
    else:wrong_ex_ph.empty()
    rep_ph.markdown(f'<div class="metric-card"><div class="metric-value">{cnt}</div><div class="metric-label">Reps</div></div>',unsafe_allow_html=True)
    stage_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem;color:{sc};">{s}</div><div class="metric-label">Stage</div></div>',unsafe_allow_html=True)
    angle_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00d4ff;">{ang}°</div><div class="metric-label">Angle</div></div>',unsafe_allow_html=True)
    fps_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.6rem;color:#f59e0b;">{gym.current_fps:.1f}</div><div class="metric-label">FPS</div></div>',unsafe_allow_html=True)
    if fb:fb_ph.markdown(f'<div class="feedback-box">💬 {fb}</div>',unsafe_allow_html=True)
    _url=f"http://localhost/movera/patient/patient-plan.php?score={sc100}"
    end_btn_ph.markdown(f'<a href="{_url}" target="_blank" style="display:block;width:100%;padding:10px 0;background:#39ff7e;color:#0d0f12;text-align:center;border-radius:10px;font-weight:700;font-size:1rem;text-decoration:none;margin-top:8px;">End Exercise (Score: {sc100})</a>',unsafe_allow_html=True)

render_stats()

# ── Webcam mode ───────────────────────────────────────────────────────────────
if mode.startswith("📹"):
    with col_vid:
        st.markdown("#### 📹 Live Webcam — Real-Time AI Tracking")

        # Check for incoming frame submitted via the hidden form
        qp = st.query_params
        if "frame" in qp:
            raw_b64 = qp["frame"]
            if raw_b64 and len(raw_b64) > 200:
                annotated = gym.process_b64(raw_b64)
                st.session_state['annotated_frame'] = annotated
                st.session_state['last_result']     = gym.result.copy()
                # Clear the query param and rerun to show result
                st.query_params.clear()
                st.rerun()

        annotated_b64 = st.session_state.get('annotated_frame','')
        ann_js = annotated_b64 if annotated_b64 else ''

        # The webcam iframe:
        #  • Captures frames from getUserMedia
        #  • Encodes to base64 JPEG
        #  • Navigates parent to ?frame=<b64>  (triggers Streamlit rerun with query param)
        #  • Also displays the last annotated frame returned via ann_js
        webcam_html = f"""
<style>
  body{{margin:0;padding:0;background:#0d0f12;font-family:sans-serif;}}
  #wrap{{width:100%;}}
  #outputCanvas{{width:100%;border-radius:10px;display:none;border:2px solid #1e2229;background:#13161c;}}
  #placeholder{{width:100%;height:360px;border-radius:10px;border:2px dashed #2a2f3a;
    background:#13161c;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:14px;color:#4b5563;font-size:.9rem;}}
  #controls{{display:flex;gap:10px;margin-top:10px;justify-content:center;flex-wrap:wrap;}}
  button{{padding:10px 24px;border-radius:8px;font-size:.9rem;font-weight:700;cursor:pointer;border:none;}}
  #startBtn{{background:#39ff7e;color:#0d0f12;}}
  #stopBtn{{background:#1e2229;color:#e8eaf0;border:1px solid #2a2f3a;display:none;}}
  #statusBar{{margin-top:8px;text-align:center;font-size:.75rem;color:#4b5563;min-height:18px;}}
  .dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
    background:#39ff7e;margin-right:5px;animation:blink 1s infinite;}}
  @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}
</style>
<div id="wrap">
  <div id="placeholder">
    <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="1.5">
      <path d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15
               14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
    </svg>
    <span>Click <b style="color:#39ff7e">Start Camera</b></span>
    <span style="font-size:.75rem;color:#374151">Allow camera access when prompted</span>
  </div>
  <canvas id="outputCanvas"></canvas>
</div>
<div id="controls">
  <button id="startBtn" onclick="startCam()">&#9654; Start Camera</button>
  <button id="stopBtn"  onclick="stopCam()">&#9632; Stop</button>
</div>
<div id="statusBar">Ready</div>
<video id="v" autoplay playsinline muted style="display:none"></video>

<script>
var video   = document.getElementById('v');
var canvas  = document.getElementById('outputCanvas');
var ctx     = canvas.getContext('2d');
var ph      = document.getElementById('placeholder');
var status  = document.getElementById('statusBar');
var stream  = null, raf = null, running = false;
var lastMs  = 0, INTERVAL = 150;  // 150ms = ~6.7 fps

// Display last annotated frame from Python (injected via f-string)
var annSrc = {repr(ann_js)};
if (annSrc && annSrc.length > 100) {{
  var img0 = new Image();
  img0.onload = function() {{
    canvas.width  = img0.naturalWidth  || 640;
    canvas.height = img0.naturalHeight || 480;
    ctx.drawImage(img0, 0, 0);
  }};
  img0.src = annSrc;
}}

function sendFrame(b64) {{
  // POST to our Tornado handler /_gym/frame
  // Falls back to query-param method if fetch fails
  fetch('/gym/frame', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/octet-stream'}},
    body: Uint8Array.from(atob(b64.split(',')[1] || b64), c => c.charCodeAt(0))
  }}).then(function() {{
    // Poll for result
    return fetch('/gym/result');
  }}).then(function(r) {{
    if (r.status === 200) return r.blob();
    return null;
  }}).then(function(blob) {{
    if (!blob) return;
    var url = URL.createObjectURL(blob);
    var img = new Image();
    img.onload = function() {{
      canvas.width  = img.naturalWidth  || 640;
      canvas.height = img.naturalHeight || 480;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
    }};
    img.src = url;
  }}).catch(function() {{
    // Tornado route injection failed — not supported on this deployment
    status.textContent = 'Live preview N/A — results shown after page refresh';
  }});
}}

function captureLoop() {{
  if (!running) return;
  raf = requestAnimationFrame(captureLoop);
  var now = performance.now();
  if (now - lastMs < INTERVAL) return;
  if (video.readyState < 2) return;
  lastMs = now;
  var off = document.createElement('canvas');
  off.width  = video.videoWidth  || 640;
  off.height = video.videoHeight || 480;
  off.getContext('2d').drawImage(video, 0, 0);
  sendFrame(off.toDataURL('image/jpeg', 0.72));
}}

window.startCam = function() {{
  status.textContent = 'Requesting camera…';
  navigator.mediaDevices.getUserMedia(
    {{video:{{width:{{ideal:640}},height:{{ideal:480}},facingMode:'user'}},audio:false}}
  ).then(function(s) {{
    stream = s; video.srcObject = stream;
    return video.play();
  }}).then(function() {{
    ph.style.display     = 'none';
    canvas.style.display = 'block';
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display  = 'inline-block';
    running = true;
    status.innerHTML = '<span class="dot"></span>Live — processing…';
    captureLoop();
  }}).catch(function(e) {{
    status.textContent = 'Camera error: ' + e.message;
  }});
}};

window.stopCam = function() {{
  running = false;
  if (raf) cancelAnimationFrame(raf);
  if (stream) stream.getTracks().forEach(function(t){{t.stop();}});
  stream = null;
  ph.style.display     = 'flex';
  canvas.style.display = 'none';
  document.getElementById('startBtn').style.display = 'inline-block';
  document.getElementById('stopBtn').style.display  = 'none';
  status.textContent = 'Stopped.';
}};
</script>
"""
        components.html(webcam_html, height=560, scrolling=False)
        render_stats()

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
                analyzer=ANALYZERS[exercise]();det_v=WrongExerciseDetector()
                prog=st.progress(0,"Processing…");preview=st.empty()
                last_res={'count':0,'stage':'','feedback':'','angle':0};fidx=0
                gym.history.add('info',f'Video started: {exercise}')
                while True:
                    ret,frame=cap.read()
                    if not ret:break
                    fidx+=1
                    det=lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),int(fidx/fps*1000))
                    if det.pose_landmarks:
                        lms=det.pose_landmarks[0]
                        if show_skeleton:draw_skeleton(frame,lms,vw,vh)
                        is_w,sus,conf=det_v.check(exercise,lms,vw,vh)
                        if is_w:draw_wrong_ex_overlay(frame,sus);gym.history.add('wrong',f'[{fidx}] {sus}')
                        last_res=analyzer.analyze(lms,vw,vh)
                    else:last_res={**last_res,'feedback':'No pose'}
                    draw_hud(frame,last_res,exercise)
                    if fidx%30==0 or fidx==tot:
                        preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True)
                        prog.progress(min(fidx/max(tot,1),1.),text=f"Frame {fidx}/{tot} — Reps:{last_res['count']}")
                cap.release();lm.close();os.unlink(tmp_path);prog.empty()
                gym.history.add('info',f'Done — {last_res["count"]} reps')
                st.success(f"✅ {last_res['count']} reps in {fidx} frames.")
                gym.result=last_res;render_stats()
                hist=list(analyzer.rc.history)
                if hist:
                    import pandas as pd
                    st.markdown("#### 📈 Angle History")
                    st.line_chart(pd.DataFrame({'angle':hist}),color="#39ff7e")
