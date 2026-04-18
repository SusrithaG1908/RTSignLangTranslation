"""
Real-Time Sign Language Translator  ·  v7.0
Streamlit  |  Light & Dark Mode  |  Montserrat font

╔══════════════════════════════════════════════════════════════════════════════╗
║  ENVIRONMENT VARIABLES                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RUN_MODE        "local" (default) | "browser"                              ║
║  CAMERA_INDEX    Camera device index for local mode. Default "0".           ║
║  RENDER          Auto-set by Render → forces browser mode                   ║
║  SPACE_ID        Auto-set by Hugging Face → forces browser mode             ║
║  PYTHON_VERSION  Set to "3.11.0" in Render env vars panel                  ║
║                                                                              ║
║  Test browser mode locally:                                                  ║
║    Windows  : set RUN_MODE=browser && streamlit run scripts/app.py          ║
║    Mac/Linux: RUN_MODE=browser streamlit run scripts/app.py                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import json
import threading
import time
import os
import importlib
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _detect_run_mode() -> str:
    if os.getenv("RENDER") == "true" or os.getenv("SPACE_ID") is not None:
        return "browser"
    return os.getenv("RUN_MODE", "local").lower()

RUN_MODE = _detect_run_mode()
IS_CLOUD = RUN_MODE == "browser"

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "word":         "",
    "history":      [],
    "last_char":    "–",
    "last_conf":    0.0,
    "stable_buf":   [],
    "run_camera":   False,
    "frame_count":  0,
    "dark_mode":    True,
    "mirror":       True,
    "run_mode":     RUN_MODE,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
#  THEME TOKENS
# ─────────────────────────────────────────────────────────────────────────────
DARK = {
    "bg":          "#1e1b3a",
    "bg2":         "#2a2540",
    "surface":     "#2e2a50",
    "surface2":    "#38346a",
    "border":      "#4a4580",
    "accent":      "#a78bfa",
    "accent_dark": "#7c5cfc",
    "accent_glow": "rgba(167,139,250,0.25)",
    "text":        "#ede9ff",
    "text_muted":  "#9d97cc",
    "success":     "#34d399",
    "warning":     "#fbbf24",
    "danger":      "#f87171",
    "card_shadow": "0 8px 32px rgba(0,0,0,0.45)",
    "btn_shadow":  "0 6px 20px rgba(167,139,250,0.35)",
    "chip_bg":     "#38346a",
    "chip_text":   "#c4b5fd",
    "live_dot":    "#34d399",
    "title_span":  "#a78bfa",
    "prog_bg":     "#38346a",
    "overlay_hex": "#a78bfa",
}
LIGHT = {
    "bg":          "#fdf6ec",
    "bg2":         "#fef9f0",
    "surface":     "#ffffff",
    "surface2":    "#fff3e0",
    "border":      "#f0d9b5",
    "accent":      "#f59e0b",
    "accent_dark": "#d97706",
    "accent_glow": "rgba(245,158,11,0.20)",
    "text":        "#1c1107",
    "text_muted":  "#8c7355",
    "success":     "#16a34a",
    "warning":     "#d97706",
    "danger":      "#dc2626",
    "card_shadow": "0 8px 32px rgba(180,120,20,0.12)",
    "btn_shadow":  "0 6px 20px rgba(245,158,11,0.30)",
    "chip_bg":     "#fff3e0",
    "chip_text":   "#92400e",
    "live_dot":    "#16a34a",
    "title_span":  "#d97706",
    "prog_bg":     "#f0d9b5",
    "overlay_hex": "#d97706",
}
T = DARK if st.session_state.dark_mode else LIGHT

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(t):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, [class*="css"] {{
    font-family: 'Montserrat', sans-serif !important;
    background-color: {t['bg']} !important;
    color: {t['text']} !important;
    transition: background-color 0.35s ease, color 0.35s ease;
}}
#MainMenu, footer, header {{ visibility: hidden !important; }}
.block-container {{ padding: 1.2rem 2rem 2rem !important; max-width: 1440px !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {t['bg2']}; }}
::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 3px; }}

/* ── TOP BAR ── */
.topbar-left {{ display: flex; align-items: center; gap: 10px; padding-top: 0.4rem; flex-wrap: wrap; }}
.app-logo {{ font-size: 2rem; line-height: 1; filter: drop-shadow(0 0 8px {t['accent_glow']}); }}
.app-name {{ font-family: 'Montserrat', sans-serif; font-size: 1.1rem; font-weight: 800; letter-spacing: -0.3px; color: {t['text']}; }}
.app-name span {{ color: {t['accent']}; }}
.env-badge {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    padding: 0.2rem 0.7rem; border-radius: 50px;
    background: {t['chip_bg']}; color: {t['chip_text']}; border: 1px solid {t['border']}; white-space: nowrap;
}}

/* ── HERO ── */
.hero {{ text-align: center; padding: 1rem 0 1.4rem; animation: fadeDown 0.55s ease both; }}
@keyframes fadeDown {{ from {{ opacity:0; transform:translateY(-16px); }} to {{ opacity:1; transform:translateY(0); }} }}
.hero h1 {{
    font-family: 'Montserrat', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.8rem); font-weight: 900;
    line-height: 1.12; letter-spacing: -1px; color: {t['text']}; margin-bottom: 0;
}}
.hero h1 span {{ color: {t['title_span']}; }}

/* ── LAYOUT ── */
.clean-card {{ background: transparent !important; padding: 0 !important; margin: 0 !important; }}

/* ── VIDEO AREA ── */
.video-off {{
    background: {t['surface2']}; border: 2px dashed {t['border']}; border-radius: 14px;
    height: 420px; width: 100%; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 0.55rem; color: {t['text_muted']};
    box-sizing: border-box;
}}
.video-off .icon {{ font-size: 3.5rem; opacity: 0.5; }}
.video-off .msg  {{ font-weight: 700; font-size: 0.95rem; }}
.video-off .sub  {{ font-size: 0.8rem; opacity: 0.65; }}

[data-testid="stImage"] img {{
    border-radius: 14px !important; width: 100% !important;
    max-height: 520px !important; object-fit: cover !important; display: block !important;
}}
div[data-testid="stCustomComponentV1"] > iframe {{
    width: 100% !important; min-height: 420px !important;
    border-radius: 14px !important; border: none !important;
}}
video {{
    width: 100% !important; max-height: 520px !important;
    border-radius: 14px !important; object-fit: cover !important;
    background: {t['bg2']};
}}

/* ── LIVE DOT ── */
.live-dot-inline {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: {t['live_dot']}; margin-right: 6px;
    animation: livePulse 1.6s ease-in-out infinite; vertical-align: middle;
}}
@keyframes livePulse {{
    0%   {{ box-shadow: 0 0 0 0   rgba(52,211,153,0.7); }}
    70%  {{ box-shadow: 0 0 0 7px rgba(52,211,153,0); }}
    100% {{ box-shadow: 0 0 0 0   rgba(52,211,153,0); }}
}}

/* ── CHARACTER PANEL ── */
.char-panel {{
    background: {t['surface2']}; border: 1px solid {t['border']}; border-radius: 16px;
    text-align: center; padding: 1.2rem 0.5rem 0.9rem; margin-bottom: 0.9rem;
    position: relative; overflow: hidden;
}}
.char-panel::before {{
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, {t['accent_glow']} 0%, transparent 68%);
    pointer-events: none;
}}
.char-panel .clabel {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.4px; color: {t['text_muted']}; margin-bottom: 0.4rem;
}}
.char-big {{
    font-family: 'Montserrat', sans-serif; font-size: 6rem; font-weight: 900;
    color: {t['accent']}; line-height: 1; text-shadow: 0 0 40px {t['accent_glow']};
    animation: charPop 0.2s cubic-bezier(.34,1.56,.64,1) both;
}}
@keyframes charPop {{ from {{ transform: scale(0.65); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}

/* ── CONFIDENCE BAR ── */
.conf-wrap {{ margin-bottom: 0.9rem; }}
.conf-row {{ display: flex; align-items: center; gap: 10px; }}
.conf-row-label {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: {t['text_muted']}; flex-shrink: 0; width: 80px; }}
.conf-track {{ flex: 1; height: 8px; background: {t['prog_bg']}; border-radius: 50px; overflow: hidden; border: 1px solid {t['border']}; }}
.conf-fill {{ height: 100%; border-radius: 50px; transition: width 0.45s cubic-bezier(.4,0,.2,1); }}
.conf-pct {{ font-size: 0.82rem; font-weight: 700; min-width: 36px; text-align: right; }}

/* ── WORD BOX ── */
.word-box {{
    background: linear-gradient(135deg, {t['accent_dark']} 0%, {t['accent']} 100%);
    border-radius: 16px; text-align: center; padding: 1.15rem 1rem 1rem; margin-bottom: 1rem;
    box-shadow: {t['btn_shadow']}; position: relative; overflow: hidden;
}}
.word-box::after {{
    content: ''; position: absolute; top: -40%; left: -10%;
    width: 80px; height: 80px; background: rgba(255,255,255,0.09);
    border-radius: 50%; filter: blur(18px); pointer-events: none;
}}
.word-box .wlabel {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.4px; color: rgba(255,255,255,0.65); margin-bottom: 0.4rem; }}
.word-text {{ font-family: 'Montserrat', sans-serif; font-size: 2.6rem; font-weight: 900; color: #fff; letter-spacing: 0.12em; min-height: 3.2rem; word-break: break-all; text-shadow: 0 2px 12px rgba(0,0,0,0.2); transition: all 0.2s ease; }}

/* ── BUTTONS ── */
.stButton > button {{
    font-family: 'Montserrat', sans-serif !important; font-weight: 700 !important;
    font-size: 0.86rem !important; border-radius: 50px !important;
    padding: 0.52rem 1.3rem !important; border: 2px solid {t['border']} !important;
    background: {t['surface']} !important; color: {t['text']} !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important; box-shadow: none !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.04) !important; box-shadow: {t['btn_shadow']} !important;
    border-color: {t['accent']} !important; color: {t['accent']} !important; background: {t['surface2']} !important;
}}
.stButton > button:active {{ transform: scale(0.97) !important; }}
div.btn-start button {{ background-color: #22c55e !important; color: white !important; border: none !important; }}
div.btn-stop  button {{ background-color: #ef4444 !important; color: white !important; border: none !important; }}

/* ── Toggles / Selects ── */
[data-testid="stToggle"] label {{ color: {t['text']} !important; font-family: 'Montserrat', sans-serif !important; font-size: 0.85rem !important; font-weight: 700 !important; }}
[data-testid="stSelectbox"] label {{ color: {t['text_muted']} !important; font-family: 'Montserrat', sans-serif !important; font-size: 0.72rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1px; }}
[data-testid="stSelectbox"] > div > div {{ background: {t['surface']} !important; border-color: {t['border']} !important; border-radius: 10px !important; color: {t['text']} !important; font-family: 'Montserrat', sans-serif !important; font-weight: 600 !important; }}

/* ── History chips ── */
.hist-wrap {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 0.4rem; }}
.hist-chip {{ background: {t['chip_bg']}; border: 1px solid {t['border']}; border-radius: 50px; padding: 0.22rem 0.8rem; font-size: 0.82rem; font-weight: 700; color: {t['chip_text']}; }}

/* ── Misc ── */
.hdivider {{ border: none; border-top: 1px solid {t['border']}; margin: 0.85rem 0; }}
.banner-warn {{ background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.38); border-radius: 12px; padding: 0.7rem 1rem; font-size: 0.82rem; color: {t['warning']}; margin-bottom: 0.9rem; }}
.fc-badge {{ text-align:center; padding-top:0.45rem; font-size:0.72rem; font-weight:700; color:{t['text_muted']}; }}
.footer {{ text-align: center; padding: 1.8rem 0 0.4rem; font-size: 0.76rem; color: {t['text_muted']}; border-top: 1px solid {t['border']}; margin-top: 2rem; }}
.footer strong {{ color: {t['accent']}; }}
[data-testid="column"] {{ padding: 0 0.5rem !important; }}
</style>
""", unsafe_allow_html=True)

inject_css(T)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "mobilenet_mp_25%_v2_best.h5")
LABELS_PATH = os.path.join("models", "class_labels_mobilenet_mp_25%_v2.json")
IMG_SIZE    = 224
CONF_THRESH = 0.70
STABILITY_N = 8

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
    ]
}

# run_mode is determined entirely by IS_CLOUD — no user selection locally

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def try_import(name):
    try:
        return importlib.import_module(name), None
    except ImportError as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────────────────────
def speak_text(text: str):
    if st.session_state.get("run_mode", RUN_MODE) == "browser":
        _speak_browser(text)
    else:
        _speak_local(text)


def _speak_browser(text: str):
    """
    Executes Web Speech API in the user's browser.
    Written into _tts_slot — a pre-allocated empty outside the columns —
    so no iframe injection disrupts the column layout.
    """
    safe = (text.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace("\n", " "))
    snippet = f"""<script>
    (function(){{
        if(!window.speechSynthesis) return;
        var u=new SpeechSynthesisUtterance('{safe}');
        u.rate=0.9; u.pitch=1.0; u.lang='en-US';
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
    }})();
    </script>"""
    try:
        _tts_slot.html(snippet)
    except Exception:
        components.html(snippet, height=0)


def _speak_local(text: str):
    def _run():
        pyttsx3, err = try_import("pyttsx3")
        if err:
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 145)
            engine.setProperty("volume", 0.95)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    tf, err = try_import("tensorflow")
    if err:
        return None, None, f"TensorFlow not found: {err}"
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model not found at `{MODEL_PATH}`"
    if not os.path.exists(LABELS_PATH):
        return None, None, f"Labels not found at `{LABELS_PATH}`"
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            first_val = next(iter(raw.values()))
            labels = ({v: k for k, v in raw.items()} if isinstance(first_val, int)
                      else {int(k): v for k, v in raw.items()})
        else:
            labels = {i: v for i, v in enumerate(raw)}
        return model, labels, None
    except Exception as e:
        return None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_hands_detector():
    mp, err = try_import("mediapipe")
    if err:
        return None, None, None, err
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
    )
    return mp, mp.solutions.drawing_utils, hands, None


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_hand_roi(frame, hand_landmarks, padding=30):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_roi(roi):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    res = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(res.astype("float32") / 255.0, 0)


def predict_character(model, labels, roi):
    if model is None or roi is None or roi.size == 0:
        return None, 0.0
    pred = model.predict(preprocess_roi(roi), verbose=0)[0]
    idx  = int(np.argmax(pred))
    return labels.get(idx, "?"), float(pred[idx])


def draw_overlay(frame_rgb, bbox, char, conf, theme):
    x1, y1, x2, y2 = bbox
    h = theme["overlay_hex"].lstrip("#")
    color = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
    label = f"{char}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame_rgb, (x1, y1 - th - 14), (x1 + tw + 14, y1), color, -1)
    cv2.putText(frame_rgb, label, (x1 + 7, y1 - 7),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_rgb


def _apply_pipeline(img, model, labels, mp_mod, mp_draw, hands_det):
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    char, conf = None, 0.0
    if model is not None and hands_det is not None:
        detection = hands_det.process(rgb)
        if detection.multi_hand_landmarks:
            hl  = detection.multi_hand_landmarks[0]
            roi, bbox = extract_hand_roi(img, hl)
            if roi.size > 0:
                char, conf = predict_character(model, labels, roi)
            mp_draw.draw_landmarks(
                rgb, hl,
                mp_mod.solutions.hands.HAND_CONNECTIONS,
                mp_mod.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp_mod.solutions.drawing_styles.get_default_hand_connections_style(),
            )
            if char and bbox:
                rgb = draw_overlay(rgb, bbox, char, conf, T)
    return rgb, char, conf


# ─────────────────────────────────────────────────────────────────────────────
#  WEBRTC PROCESSOR  (cloud browser mode only)
# ─────────────────────────────────────────────────────────────────────────────
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.labels, _ = load_model_and_labels()
        self.mp_mod, self.mp_draw, self.hands_det, _ = get_hands_detector()
        self._stable_buf: list[str] = []
        self._word  = ""
        self._lock  = threading.Lock()
        self.result = {"char": "–", "conf": 0.0, "word": ""}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if st.session_state.get("mirror", True):
            img = cv2.flip(img, 1)
        rgb, char, conf = _apply_pipeline(
            img, self.model, self.labels,
            self.mp_mod, self.mp_draw, self.hands_det,
        )
        with self._lock:
            if char and conf >= CONF_THRESH:
                self._stable_buf.append(char)
                if len(self._stable_buf) > STABILITY_N:
                    self._stable_buf.pop(0)
                if (len(self._stable_buf) == STABILITY_N
                        and len(set(self._stable_buf)) == 1):
                    if not self._word or self._word[-1] != char:
                        self._word += char
                    self._stable_buf.clear()
                self.result = {"char": char, "conf": conf, "word": self._word}
            else:
                if self._stable_buf:
                    self._stable_buf.pop(0)
                self.result = {"char": char or "–", "conf": conf, "word": self._word}
        return av.VideoFrame.from_ndarray(rgb, format="rgb24")

    def reset_word(self):
        with self._lock:
            self._word = ""
            self._stable_buf.clear()
            self.result = {"char": "–", "conf": 0.0, "word": ""}

    def delete_last(self):
        with self._lock:
            self._word = self._word[:-1]
            self.result["word"] = self._word


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
model, labels, model_err           = load_model_and_labels()
mp_mod, mp_draw, hands_det, mp_err = get_hands_detector()


# ─────────────────────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
if IS_CLOUD:
    st.session_state.run_mode = "browser"

_active_mode = st.session_state.get("run_mode", RUN_MODE)
if _active_mode == "browser":
    _env_label = "☁️ Cloud · WebRTC"
else:
    _env_label = "💻 Local · OpenCV"

tb_l, tb_r = st.columns([3, 1])
with tb_l:
    st.markdown(f"""
    <div class="topbar-left">
        <span class="app-logo">🤟</span>
        <span class="app-name">Sign<span>Lang</span> AI</span>
        <span class="env-badge">{_env_label}</span>
    </div>""", unsafe_allow_html=True)
with tb_r:
    tr1, tr2 = st.columns([1, 1])
    with tr1:
        st.session_state.mirror = st.toggle(
            "🪞 Mirror", value=st.session_state.mirror, key="top_mirror")
    with tr2:
        _dark_lbl = "🌙 Dark" if st.session_state.dark_mode else "🌞 Light"
        _new_dark = st.toggle(_dark_lbl, value=st.session_state.dark_mode, key="theme_toggle")
        if _new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = _new_dark
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Real-Time <span>Sign Language</span> Translator</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ERROR BANNERS
# ─────────────────────────────────────────────────────────────────────────────
if model_err:
    st.markdown(f'<div class="banner-warn">⚠️ Model: {model_err}</div>', unsafe_allow_html=True)
if mp_err:
    st.markdown(f'<div class="banner-warn">⚠️ MediaPipe: {mp_err}</div>', unsafe_allow_html=True)
if model_err or mp_err:
    st.markdown("""<div class="banner-warn">
    💡 <b>Demo mode active.</b> Place model files in <code>models/</code> and restart.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
# _tts_slot: pre-declared outside columns so speak_text() can write into it
# without disrupting column layout. Renders zero pixels.
_tts_slot = st.empty()

col_vid, col_out = st.columns([1.15, 0.85], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT — output panel
# ══════════════════════════════════════════════════════════════════════════════
with col_out:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)

    char_slot = st.empty()
    conf_slot = st.empty()

    def render_char_conf(char, conf):
        char_slot.markdown(f"""
        <div class="char-panel">
            <div class="clabel">Detected Character</div>
            <div class="char-big">{char}</div>
        </div>""", unsafe_allow_html=True)
        pct  = conf * 100
        fill = (T["success"] if conf >= 0.80
                else (T["warning"] if conf >= CONF_THRESH else T["danger"]))
        conf_slot.markdown(f"""
        <div class="conf-wrap">
          <div class="conf-row">
            <span class="conf-row-label">Confidence</span>
            <div class="conf-track">
              <div class="conf-fill" style="width:{pct:.1f}%;background:{fill};"></div>
            </div>
            <span class="conf-pct" style="color:{fill};">{pct:.0f}%</span>
          </div>
        </div>""", unsafe_allow_html=True)

    word_slot = st.empty()

    def render_word(word):
        display = word if word else "…"
        word_slot.markdown(f"""
        <div class="word-box">
            <div class="wlabel">Formed Word</div>
            <div class="word-text">{display}</div>
        </div>""", unsafe_allow_html=True)

    render_char_conf(st.session_state.last_char, st.session_state.last_conf)
    render_word(st.session_state.word)

    # Action buttons
    ab1, ab2, ab3 = st.columns(3)
    with ab1:
        if st.button("🔊 Speak", use_container_width=True, key="btn_speak"):
            if st.session_state.word:
                speak_text(st.session_state.word)
            else:
                st.toast("Nothing to speak yet!", icon="🤫")
    with ab2:
        if st.button("⌫ Delete", use_container_width=True, key="btn_back"):
            st.session_state.word = st.session_state.word[:-1]
            render_word(st.session_state.word)
    with ab3:
        if st.button("🔄 Reset", use_container_width=True, key="btn_reset"):
            if st.session_state.word:
                st.session_state.history.append(st.session_state.word)
            st.session_state.word       = ""
            st.session_state.stable_buf = []
            st.session_state.last_char  = "–"
            st.session_state.last_conf  = 0.0
            st.session_state.run_camera = False
            st.rerun()

    if st.session_state.history:
        st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
        chips = "".join(
            f'<span class="hist-chip">{w}</span>'
            for w in st.session_state.history[-6:]
        )
        st.markdown(f'<div class="hist-wrap">{chips}</div>', unsafe_allow_html=True)

    # Settings — only shown locally, only one mode available (OpenCV)
    # Browser/WebRTC mode is cloud-only and activates automatically via IS_CLOUD.
    if not IS_CLOUD:
        st.session_state.run_mode = "local"   # enforce local mode on local machine

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT — video panel  (display only — no buttons, no controls)
# ══════════════════════════════════════════════════════════════════════════════
with col_vid:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)

    active_mode = st.session_state.run_mode

    if active_mode == "browser":
        # Cloud-only: WebRTC streaming
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:700;color:{T["text_muted"]};'
            f'margin-bottom:0.5rem;"><span class="live-dot-inline"></span>'
            f'WebRTC · Browser Camera · Web Speech TTS</div>',
            unsafe_allow_html=True,
        )
        ctx = webrtc_streamer(
            key="sign-lang-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                "audio": False,
            },
            async_processing=True,
            translations={
                "start": "▶ Start Camera",
                "stop":  "⏹ Stop Camera",
            },
        )
        if ctx.state.playing and ctx.video_processor:
            while ctx.state.playing:
                result = ctx.video_processor.result
                char = result.get("char", "–")
                conf = result.get("conf", 0.0)
                word = result.get("word", "")
                st.session_state.last_char = char
                st.session_state.last_conf = conf
                st.session_state.word      = word
                render_char_conf(char, conf)
                render_word(word)
                time.sleep(0.10)

    else:
        # Local OpenCV mode — mirrors snapshot mode structure exactly:
        #   label row  →  Start/Stop buttons  →  video area
        #
        # video_slot is declared HERE inside the column so it occupies the
        # correct visual position. The camera loop (after footer) holds the
        # same reference and writes frames into it — st.empty() references
        # remain valid and writable from anywhere in the script.

        # Status label — mirrors snapshot mode's label exactly
        dot = '<span class="live-dot-inline"></span>' if st.session_state.run_camera else ""
        status = "Live · OpenCV · pyttsx3 TTS" if st.session_state.run_camera else "OpenCV · pyttsx3 TTS"
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:700;color:{T["text_muted"]};'
            f'margin-bottom:0.5rem;">{dot}{status}</div>',
            unsafe_allow_html=True,
        )

        # Video area — declared inside the column so it sits in the right place.
        # The camera loop below writes frames here via this reference.
        video_slot = st.empty()
        if not st.session_state.run_camera:
            video_slot.markdown("""
            <div class="video-off">
                <div class="icon">📷</div>
                <div class="msg">Camera is off</div>
                <div class="sub">Press ▶ Start to begin detection</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Write a loading placeholder so the slot has height and is visible
            # in the column. The camera loop overwrites this with real frames.
            video_slot.markdown("""
            <div class="video-off">
                <div class="icon">⏳</div>
                <div class="msg">Starting camera…</div>
                <div class="sub">First frame loading</div>
            </div>""", unsafe_allow_html=True)

        # Start / Stop row — below video
        st.markdown('<div style="margin-top:0.6rem;"></div>', unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns([1, 1, 1])
        with vc1:
            st.markdown('<div class="btn-start">', unsafe_allow_html=True)
            if st.button("▶ Start", use_container_width=True,
                         disabled=st.session_state.run_camera, key="btn_start"):
                st.session_state.run_camera  = True
                st.session_state.frame_count = 0
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with vc2:
            st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not st.session_state.run_camera, key="btn_stop"):
                st.session_state.run_camera = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with vc3:
            fc  = st.session_state.get("frame_count", 0)
            txt = ("🎞 " + str(fc)) if fc else "⏳ Ready"
            st.markdown(f'<div class="fc-badge">{txt}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
_mode = st.session_state.run_mode
tts_tech = "Web Speech API" if _mode == "browser" else "pyttsx3"
cam_tech = "WebRTC" if _mode == "browser" else "OpenCV"
st.markdown(f"""
<div class="footer">
    Built with 🤟 using
    <strong>Streamlit</strong> ·
    <strong>TensorFlow 2.15</strong> ·
    <strong>MediaPipe</strong> ·
    <strong>OpenCV</strong> ·
    <strong>{cam_tech}</strong> ·
    <strong>{tts_tech}</strong>
    &nbsp;—&nbsp; Sign Language Translator v7.0
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL CAMERA LOOP
#  Runs AFTER all columns have rendered — including Start/Stop buttons which
#  are now inside col_vid above the video area.
#  video_slot was declared inside col_vid; its reference is valid here.
#  Stop sets run_camera=False → next rerun skips this block entirely.
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run_mode == "local" and st.session_state.run_camera:
    cam_idx = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        with col_vid:
            st.error(f"❌ Could not open camera {cam_idx}. Check Camera Device selection.")
        st.session_state.run_camera = False
        st.stop()

    try:
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if st.session_state.mirror:
                frame = cv2.flip(frame, 1)

            st.session_state.frame_count += 1

            rgb, char, conf = _apply_pipeline(
                frame, model, labels, mp_mod, mp_draw, hands_det,
            )

            with col_vid:
                video_slot.image(rgb, channels="RGB", use_container_width=True)

            if char and conf >= CONF_THRESH:
                st.session_state.last_char = char
                st.session_state.last_conf = conf
                buf = st.session_state.stable_buf
                buf.append(char)
                if len(buf) > STABILITY_N:
                    buf.pop(0)
                if len(buf) == STABILITY_N and len(set(buf)) == 1:
                    if not st.session_state.word or st.session_state.word[-1] != char:
                        st.session_state.word += char
                    buf.clear()
            else:
                if st.session_state.stable_buf:
                    st.session_state.stable_buf.pop(0)
                if not char:
                    st.session_state.last_char = "–"
                    st.session_state.last_conf = 0.0

            render_char_conf(st.session_state.last_char, st.session_state.last_conf)
            render_word(st.session_state.word)
            time.sleep(0.05)
    finally:
        cap.release()