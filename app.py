"""
Real-Time Sign Language Translator  ·  v8.2
Streamlit  |  Light & Dark Mode  |  Montserrat font

╔══════════════════════════════════════════════════════════════════════════════╗
║  ENVIRONMENT VARIABLES                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RUN_MODE        "local" (default) | "browser"                              ║
║  CAMERA_INDEX    Camera device index for local mode. Default "0".           ║
║  RENDER          Auto-set by Render → forces browser mode                   ║
║  SPACE_ID        Auto-set by Hugging Face → forces browser mode             ║
║                                                                              ║
║  Architecture (v8.1):                                                        ║
║    LOCAL  : OpenCV loop → st.image()                                        ║
║    CLOUD  : Browser JS captures webcam → POST /predict (FastAPI)            ║
║             → MediaPipe + TF inference → JSON response → UI update          ║
║             No WebRTC, no TURN servers, no ICE negotiation.                 ║
║                                                                              ║
║  Cloud simulation (v8.1 new):                                                ║
║    When running locally a "☁ Cloud" toggle appears in the top bar.          ║
║    Switching it on starts FastAPI on port 8000 and activates the browser    ║
║    JS camera widget — exactly as it runs on Hugging Face — so you can       ║
║    fully test the cloud path before deploying.                              ║
║    On real cloud (SPACE_ID / RENDER set) the toggle is hidden and browser   ║
║    mode is always active.                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os

# ── MUST be before ANY other imports ──────────────────────────────────────────
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import importlib
import json
import threading
import time

import base64
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT DETECTION
#  IS_CLOUD  → True only on real deployment (env var set by HF / Render).
#              The UI toggle is hidden; browser mode is locked on.
#  IS_CLOUD  → False on local machine.
#              A "☁ Cloud" toggle lets you switch to browser/HTTP mode for
#              pre-deployment testing without restarting Streamlit.
# ─────────────────────────────────────────────────────────────────────────────
def _detect_run_mode() -> str:
    if os.getenv("RENDER") == "true" or os.getenv("SPACE_ID") is not None:
        return "browser"
    return os.getenv("RUN_MODE", "local").lower()

RUN_MODE = _detect_run_mode()
IS_CLOUD = RUN_MODE == "browser"   # True only on real cloud deployment

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICT HTTP SERVER
#
#  Uses Python's built-in http.server — zero extra dependencies, no uvicorn,
#  nothing that can be auto-discovered or accidentally launched by the venv.
#
#  Listens on port 8000. Started in a daemon thread only when:
#    • IS_CLOUD is True  (real HF/Render deployment — started immediately)
#    • ☁ Cloud toggle is turned on locally (started on demand, once)
#
#  _api_* module-level references are filled after model/mediapipe load below.
# ─────────────────────────────────────────────────────────────────────────────
_api_model     = None
_api_labels    = None
_api_mp_mod    = None
_api_mp_draw   = None
_api_hands_det = None

_predict_server_started = threading.Event()   # guard: start at most once

def _ensure_predict_server_running():
    """
    Start a lightweight HTTP predict server using only stdlib http.server.
    No uvicorn, no FastAPI, no ASGI — nothing that can be auto-discovered.
    Safe to call multiple times — only the first call does any work.
    """
    if _predict_server_started.is_set():
        return
    _predict_server_started.set()

    import json as _json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _PredictHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress request logs

        def do_OPTIONS(self):
            self.send_response(200)
            self._cors()
            self.end_headers()

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = _json.loads(self.rfile.read(length))
                img_bytes = base64.b64decode(body["image"])
                arr       = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Could not decode image")
                if body.get("mirror", False):
                    frame = cv2.flip(frame, 1)
                _, char, conf = _apply_pipeline(
                    frame,
                    _api_model, _api_labels,
                    _api_mp_mod, _api_mp_draw, _api_hands_det,
                )
                result = {"char": char or "–", "conf": round(float(conf), 3)}
            except Exception as exc:
                result = {"char": "–", "conf": 0.0, "error": str(exc)}

            payload = _json.dumps(result).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _run():
        server = HTTPServer(("0.0.0.0", 8000), _PredictHandler)
        server.serve_forever()

    threading.Thread(target=_run, daemon=True).start()

# On real cloud deployment start the server before the first Streamlit render
if IS_CLOUD:
    _ensure_predict_server_running()

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
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "word":           "",
    "history":        [],
    "last_char":      "–",
    "last_conf":      0.0,
    "stable_buf":     [],
    "run_camera":     False,
    "frame_count":    0,
    "dark_mode":      True,
    "mirror":         True,
    "run_mode":       RUN_MODE,
    # simulate_cloud: local-only toggle; always False on real cloud
    "simulate_cloud": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Derive the active mode for this render cycle:
#   real cloud              → always "browser"
#   local + sim toggle on   → "browser"
#   local + sim toggle off  → "local"
ACTIVE_MODE = "browser" if (IS_CLOUD or st.session_state.simulate_cloud) else "local"
st.session_state.run_mode = ACTIVE_MODE

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
.sim-badge {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    padding: 0.2rem 0.7rem; border-radius: 50px;
    background: rgba(251,191,36,0.15); color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.45); white-space: nowrap;
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
.banner-info {{ background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.30); border-radius: 12px; padding: 0.7rem 1rem; font-size: 0.82rem; color: {t['accent']}; margin-bottom: 0.9rem; }}
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
    if ACTIVE_MODE == "browser":
        _speak_browser(text)
    else:
        _speak_local(text)


def _speak_browser(text: str):
    """Inject Web Speech API into _tts_slot (zero-height, outside columns)."""
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
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0 if IS_CLOUD else 1,
    )
    return mp, mp.solutions.drawing_utils, hands, None


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION PIPELINE
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
    """Run MediaPipe + model on a BGR frame. Used by both modes."""
    rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    char, conf = None, 0.0
    if model is not None and hands_det is not None:
        detection = hands_det.process(rgb)
        if detection.multi_hand_landmarks:
            hl = detection.multi_hand_landmarks[0]
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
#  LOAD RESOURCES  &  wire into FastAPI endpoint
# ─────────────────────────────────────────────────────────────────────────────
model, labels, model_err           = load_model_and_labels()
mp_mod, mp_draw, hands_det, mp_err = get_hands_detector()

_api_model     = model
_api_labels    = labels
_api_mp_mod    = mp_mod
_api_mp_draw   = mp_draw
_api_hands_det = hands_det

# ─────────────────────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
if IS_CLOUD:
    _env_label   = "☁️ Cloud · HTTP Fetch"
    _badge_class = "env-badge"
elif st.session_state.simulate_cloud:
    _env_label   = "🧪 Simulated Cloud · HTTP Fetch"
    _badge_class = "sim-badge"
else:
    _env_label   = "💻 Local · OpenCV"
    _badge_class = "env-badge"

tb_l, tb_r = st.columns([3, 1])
with tb_l:
    st.markdown(f"""
    <div class="topbar-left">
        <span class="app-logo">🤟</span>
        <span class="app-name">Sign<span>Lang</span> AI</span>
        <span class="{_badge_class}">{_env_label}</span>
    </div>""", unsafe_allow_html=True)

with tb_r:
    # On real cloud: Mirror + Dark/Light only (2 columns)
    # Locally:       ☁ Cloud + Mirror + Dark/Light (3 columns)
    if not IS_CLOUD:
        tr0, tr1, tr2 = st.columns([1.2, 1, 1])
        with tr0:
            prev_sim = st.session_state.simulate_cloud
            new_sim  = st.toggle(
                "☁ Cloud", value=prev_sim, key="sim_toggle",
                help="Simulate Hugging Face cloud mode locally.\n"
                     "Starts FastAPI on :8000 and switches to the\n"
                     "browser JS camera widget instead of OpenCV.\n"
                     "Toggle off to return to local OpenCV mode.",
            )
            if new_sim != prev_sim:
                st.session_state.simulate_cloud = new_sim
                # Reset all camera/word state so neither mode inherits stale values
                st.session_state.run_camera  = False
                st.session_state.word        = ""
                st.session_state.stable_buf  = []
                st.session_state.last_char   = "–"
                st.session_state.last_conf   = 0.0
                st.session_state.frame_count = 0
                if new_sim:
                    _ensure_predict_server_running()
                st.rerun()
    else:
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
#  INFO BANNER  (local cloud-simulation only)
# ─────────────────────────────────────────────────────────────────────────────
if not IS_CLOUD and st.session_state.simulate_cloud:
    st.markdown(
        '<div class="banner-info">🧪 <b>Cloud simulation active.</b> '
        'FastAPI is running on <code>localhost:8000</code>. '
        'The browser JS camera widget is POSTing frames to <code>/predict</code> '
        'exactly as it will on Hugging Face. '
        'Toggle <b>☁ Cloud</b> off to return to local OpenCV mode.</div>',
        unsafe_allow_html=True,
    )

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
# _tts_slot: zero-height placeholder outside columns for browser TTS injection
_tts_slot = st.empty()

col_vid, col_out = st.columns([1.15, 0.85], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT — output panel  (shared by both modes)
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

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT — video panel
# ══════════════════════════════════════════════════════════════════════════════
with col_vid:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)

    # ── BROWSER MODE (real cloud or local simulation) ─────────────────────────
    if ACTIVE_MODE == "browser":
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:700;color:{T["text_muted"]};'
            f'margin-bottom:0.5rem;"><span class="live-dot-inline"></span>'
            f'Browser Camera · HTTP Fetch · Web Speech TTS</div>',
            unsafe_allow_html=True,
        )

        mirror_js = "true" if st.session_state.mirror else "false"
        accent    = T["accent"]
        bg2       = T["bg2"]
        text_c    = T["text"]
        muted     = T["text_muted"]
        success   = T["success"]
        warning   = T["warning"]
        danger    = T["danger"]
        border    = T["border"]

        browser_widget = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'Montserrat', sans-serif; }}
  #wrap {{ position: relative; width: 100%; }}
  video {{
    width: 100%; border-radius: 14px; display: block;
    background: {bg2}; max-height: 420px; object-fit: cover;
  }}
  #overlay {{
    position: absolute; top: 10px; left: 10px;
    background: rgba(0,0,0,0.55); border-radius: 10px;
    padding: 6px 12px; font-size: 1.1rem; font-weight: 800;
    color: #fff; display: none;
  }}
  #status {{
    margin-top: 8px; font-size: 0.75rem; font-weight: 700;
    color: {muted}; text-align: center;
  }}
  #btn {{
    display: block; margin: 10px auto 0; padding: 0.5rem 2rem;
    border-radius: 50px; border: 2px solid {border};
    background: transparent; color: {text_c};
    font-family: 'Montserrat', sans-serif; font-weight: 700;
    font-size: 0.86rem; cursor: pointer; transition: all 0.2s;
  }}
  #btn.running {{ background: #ef4444; border-color: #ef4444; color: #fff; }}
  #btn:hover {{ border-color: {accent}; color: {accent}; }}
  #btn.running:hover {{ background: #dc2626; border-color: #dc2626; color: #fff; }}
</style>
</head>
<body>
<div id="wrap">
  <video id="cam" autoplay playsinline muted></video>
  <canvas id="canvas" style="display:none" width="320" height="240"></canvas>
  <div id="overlay"></div>
</div>
<div id="status">Click ▶ Start Camera to begin</div>
<button id="btn" onclick="toggleCamera()">▶ Start Camera</button>

<script>
const video   = document.getElementById('cam');
const canvas  = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const status  = document.getElementById('status');
const btn     = document.getElementById('btn');
const ctx     = canvas.getContext('2d');
const MIRROR  = {mirror_js};
const THRESH  = 0.70;
const STAB_N  = 8;

let stableBuf  = [];
let word       = '';
let running    = false;
let stream     = null;
let intervalId = null;
let framesSent = 0;

function toggleCamera() {{
  if (!running) startCamera(); else stopCamera();
}}

async function startCamera() {{
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ width: 640, height: 480, facingMode: 'user' }},
      audio: false,
    }});
    video.srcObject = stream;
    await video.play();
    running = true;
    btn.textContent = '⏹ Stop Camera';
    btn.classList.add('running');
    status.textContent = 'Camera live — detecting…';
    overlay.style.display = 'block';
    intervalId = setInterval(sendFrame, 200);   // 5 fps
  }} catch(e) {{
    status.textContent = '❌ Camera error: ' + e.message;
  }}
}}

function stopCamera() {{
  clearInterval(intervalId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  running = false;
  btn.textContent = '▶ Start Camera';
  btn.classList.remove('running');
  overlay.style.display = 'none';
  status.textContent = 'Camera stopped.';
  framesSent = 0;
}}

async function sendFrame() {{
  if (!running || video.readyState < 2) return;

  ctx.save();
  if (MIRROR) {{
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }}
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  const b64 = canvas.toDataURL('image/jpeg', 0.75).split(',')[1];

  try {{
    const resp = await fetch('http://localhost:8000/predict', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ image: b64, mirror: false }}),
    }});
    if (!resp.ok) return;
    const data = await resp.json();
    framesSent++;
    processResult(data.char, data.conf);
    status.textContent = `Frames sent: ${{framesSent}} · Last: ${{data.char}} (${{Math.round(data.conf*100)}}%)`;
  }} catch(e) {{
    status.textContent = '⚠️ Backend unreachable — retrying…';
  }}
}}

function processResult(char, conf) {{
  if (char && char !== '–' && conf >= THRESH) {{
    overlay.textContent = char + '  ' + Math.round(conf * 100) + '%';
    overlay.style.color = conf >= 0.80 ? '{success}' : (conf >= THRESH ? '{warning}' : '{danger}');
    stableBuf.push(char);
    if (stableBuf.length > STAB_N) stableBuf.shift();
    if (stableBuf.length === STAB_N && new Set(stableBuf).size === 1) {{
      if (!word || word[word.length-1] !== char) word += char;
      stableBuf = [];
    }}
  }} else {{
    if (stableBuf.length) stableBuf.shift();
    overlay.textContent = char === '–' ? '' : (char || '');
  }}
  window.parent.postMessage({{
    type: 'signlang',
    char: (char && conf >= THRESH) ? char : '–',
    conf: conf,
    word: word,
  }}, '*');
}}
</script>
</body>
</html>
"""
        components.html(browser_widget, height=520, scrolling=False)

        st.markdown("""
<script>
window.addEventListener('message', function(e) {
  if (!e.data || e.data.type !== 'signlang') return;
  const p = new URLSearchParams(window.location.search);
  p.set('sl_char', e.data.char  || '–');
  p.set('sl_conf', (e.data.conf || 0).toFixed(3));
  p.set('sl_word', e.data.word  || '');
  window.history.replaceState({}, '', '?' + p.toString());
}, false);
</script>
""", unsafe_allow_html=True)

        qp      = st.query_params
        sl_char = qp.get("sl_char", st.session_state.last_char)
        sl_conf = float(qp.get("sl_conf", st.session_state.last_conf))
        sl_word = qp.get("sl_word", st.session_state.word)

        if sl_char != st.session_state.last_char or sl_word != st.session_state.word:
            st.session_state.last_char = sl_char
            st.session_state.last_conf = sl_conf
            st.session_state.word      = sl_word
            render_char_conf(sl_char, sl_conf)
            render_word(sl_word)

    # ── LOCAL MODE: unchanged OpenCV loop ─────────────────────────────────────
    else:
        dot    = '<span class="live-dot-inline"></span>' if st.session_state.run_camera else ""
        status = "Live · OpenCV · pyttsx3 TTS" if st.session_state.run_camera else "OpenCV · pyttsx3 TTS"
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:700;color:{T["text_muted"]};'
            f'margin-bottom:0.5rem;">{dot}{status}</div>',
            unsafe_allow_html=True,
        )

        video_slot = st.empty()
        if not st.session_state.run_camera:
            video_slot.markdown("""
            <div class="video-off">
                <div class="icon">📷</div>
                <div class="msg">Camera is off</div>
                <div class="sub">Press ▶ Start to begin detection</div>
            </div>""", unsafe_allow_html=True)
        else:
            video_slot.markdown("""
            <div class="video-off">
                <div class="icon">⏳</div>
                <div class="msg">Starting camera…</div>
                <div class="sub">First frame loading</div>
            </div>""", unsafe_allow_html=True)

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
tts_tech = "Web Speech API" if ACTIVE_MODE == "browser" else "pyttsx3"
cam_tech = "HTTP Fetch (no WebRTC)" if ACTIVE_MODE == "browser" else "OpenCV"
st.markdown(f"""
<div class="footer">
    Built with 🤟 using
    <strong>Streamlit</strong> ·
    <strong>TensorFlow 2.15</strong> ·
    <strong>MediaPipe</strong> ·
    <strong>OpenCV</strong> ·
    <strong>{cam_tech}</strong> ·
    <strong>{tts_tech}</strong>
    &nbsp;—&nbsp; Sign Language Translator v8.1
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL CAMERA LOOP  (runs only when ACTIVE_MODE == "local")
#
#  video_slot is declared inside col_vid above. Its reference stays valid here.
#  Stop sets run_camera=False → next rerun skips this block entirely.
# ─────────────────────────────────────────────────────────────────────────────
if ACTIVE_MODE == "local" and st.session_state.run_camera:
    cam_idx = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        with col_vid:
            st.error(f"❌ Could not open camera {cam_idx}. Check CAMERA_INDEX env var.")
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