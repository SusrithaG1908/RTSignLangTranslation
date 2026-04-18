

"""
Real-Time Sign Language Translator  ·  v3.0
Streamlit  |  Light & Dark Mode  |  Montserrat font
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
import threading
import os
import importlib

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
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
    "word":        "",
    "history":     [],
    "last_char":   "–",
    "last_conf":   0.0,
    "stable_buf":  [],
    "run_camera":  False,
    "frame_count": 0,
    "dark_mode":   True,
    "mirror":      True,
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
/* Montserrat replaces Outfit/Syne everywhere */
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
.topbar-left {{ display: flex; align-items: center; gap: 10px; padding-top: 0.4rem; }}
.app-logo {{ font-size: 2rem; line-height: 1; filter: drop-shadow(0 0 8px {t['accent_glow']}); }}
.app-name {{
    font-family: 'Montserrat', sans-serif; font-size: 1.1rem; font-weight: 800;
    letter-spacing: -0.3px; color: {t['text']};
}}
.app-name span {{ color: {t['accent']}; }}

/* ── HERO — no badge, no subtitle ── */
.hero {{
    text-align: center; padding: 1.2rem 0 1.8rem;
    animation: fadeDown 0.55s ease both;
}}
@keyframes fadeDown {{
    from {{ opacity:0; transform:translateY(-16px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.hero h1 {{
    font-family: 'Montserrat', sans-serif;
    font-size: clamp(1.9rem, 4vw, 3.1rem); font-weight: 900;
    line-height: 1.12; letter-spacing: -1px; color: {t['text']}; margin-bottom: 0;
}}
.hero h1 span {{ color: {t['title_span']}; }}

/* ── CARD ── */
.card {{
    background: transparent !important;   /* 🔥 removes purple block */
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(14px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

.clean-card {{
    background: {t['surface']};
    border-radius: 16px;
    padding: 1rem;
}}

.clean-card {{
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}}
.card-title {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: {t['text_muted']};
}}

/* ── LIVE DOT (kept for the card-title area only) ── */
.live-dot-inline {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: {t['live_dot']}; margin-right: 6px;
    animation: livePulse 1.6s ease-in-out infinite;
    vertical-align: middle;
}}
@keyframes livePulse {{
    0%   {{ box-shadow: 0 0 0 0   rgba(52,211,153,0.7); }}
    70%  {{ box-shadow: 0 0 0 7px rgba(52,211,153,0); }}
    100% {{ box-shadow: 0 0 0 0   rgba(52,211,153,0); }}
}}

/* ── VIDEO OFF PLACEHOLDER ── */
.video-off {{
    background: {t['surface2']}; border: 2px dashed {t['border']}; border-radius: 14px;
    height: 340px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 0.55rem; color: {t['text_muted']};
}}
.video-off .icon {{ font-size: 3.5rem; opacity: 0.5; }}
.video-off .msg  {{ font-weight: 700; font-size: 0.95rem; }}
.video-off .sub  {{ font-size: 0.8rem; opacity: 0.65; }}
[data-testid="stImage"] img {{ border-radius: 14px !important; width: 100% !important; display: block; }}

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
@keyframes charPop {{
    from {{ transform: scale(0.65); opacity: 0; }}
    to   {{ transform: scale(1);    opacity: 1; }}
}}

/* ── CONFIDENCE BAR ── */
.conf-wrap {{ margin-bottom: 0.9rem; }}
.conf-row {{ display: flex; align-items: center; gap: 10px; }}
.conf-row-label {{
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: {t['text_muted']}; flex-shrink: 0; width: 80px;
}}
.conf-track {{
    flex: 1; height: 8px; background: {t['prog_bg']};
    border-radius: 50px; overflow: hidden; border: 1px solid {t['border']};
}}
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
.word-box .wlabel {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.4px; color: rgba(255,255,255,0.65); margin-bottom: 0.4rem;
}}
.word-text {{
    font-family: 'Montserrat', sans-serif; font-size: 2.6rem; font-weight: 900;
    color: #fff; letter-spacing: 0.12em; min-height: 3.2rem;
    word-break: break-all; text-shadow: 0 2px 12px rgba(0,0,0,0.2); transition: all 0.2s ease;
}}

/* ── BUTTONS — base style ── */
.stButton > button {{
    font-family: 'Montserrat', sans-serif !important; font-weight: 700 !important;
    font-size: 0.86rem !important; border-radius: 50px !important;
    padding: 0.52rem 1.3rem !important;
    border: 2px solid {t['border']} !important;
    background: {t['surface']} !important; color: {t['text']} !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important; box-shadow: none !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.04) !important;
    box-shadow: {t['btn_shadow']} !important;
    border-color: {t['accent']} !important; color: {t['accent']} !important;
    background: {t['surface2']} !important;
}}
.stButton > button:active {{ transform: scale(0.97) !important; }}

/* START BUTTON - FORCE GREEN */
div.btn-start button[kind="secondary"],
div.btn-start button {{
    background-color: #22c55e !important;
    color: white !important;
    border: none !important;
}}

/* STOP BUTTON - FORCE RED */
div.btn-stop button[kind="secondary"],
div.btn-stop button {{
    background-color: #ef4444 !important;
    color: white !important;
    border: none !important;
}}
/* ── Toggle / Select ── */
[data-testid="stToggle"] label {{
    color: {t['text']} !important; font-family: 'Montserrat', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 700 !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {t['surface']} !important; border-color: {t['border']} !important;
    border-radius: 12px !important; color: {t['text']} !important;
    font-family: 'Montserrat', sans-serif !important;
}}

/* ── History chips ── */
.hist-wrap {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 0.4rem; }}
.hist-chip {{
    background: {t['chip_bg']}; border: 1px solid {t['border']}; border-radius: 50px;
    padding: 0.22rem 0.8rem; font-size: 0.82rem; font-weight: 700; color: {t['chip_text']};
    transition: transform 0.15s ease; cursor: default;
}}
.hist-chip:hover {{ transform: scale(1.07); }}

/* ── Misc ── */
.hdivider {{ border: none; border-top: 1px solid {t['border']}; margin: 0.85rem 0; }}
.banner-warn {{
    background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.38);
    border-radius: 12px; padding: 0.7rem 1rem; font-size: 0.82rem;
    color: {t['warning']}; margin-bottom: 0.9rem;
}}
.footer {{
    text-align: center; padding: 1.8rem 0 0.4rem; font-size: 0.76rem;
    color: {t['text_muted']}; border-top: 1px solid {t['border']}; margin-top: 2rem;
}}
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


def try_import(name):
    try:
        return importlib.import_module(name), None
    except ImportError as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
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
            if isinstance(first_val, int):
                labels = {v: k for k, v in raw.items()}
            else:
                labels = {int(k): v for k, v in raw.items()}
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
    conf = float(pred[idx])
    return labels.get(idx, "?"), conf


def draw_overlay(frame_rgb, bbox, char, conf, theme):
    x1, y1, x2, y2 = bbox
    h = theme["overlay_hex"].lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    color = (r, g, b)
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
    label = f"{char}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame_rgb, (x1, y1 - th - 14), (x1 + tw + 14, y1), color, -1)
    cv2.putText(frame_rgb, label, (x1 + 7, y1 - 7),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_rgb


# ─────────────────────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────────────────────
def speak_text(text: str):
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
#  LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
model, labels, model_err = load_model_and_labels()
mp_mod, mp_draw, hands_det, mp_err = get_hands_detector()
resource_ok = (model is not None) and (hands_det is not None)


# ─────────────────────────────────────────────────────────────────────────────
#  TOP BAR  (logo + theme toggle)
# ─────────────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown("""
    <div class="topbar-left">
        <span class="app-logo">🤟</span>
        <span class="app-name">Sign<span>Lang</span> AI</span>
    </div>
    """, unsafe_allow_html=True)

with top_r:
    toggle_label = "🌙 Dark Mode" if st.session_state.dark_mode else "🌞 Light Mode"
    new_dark = st.toggle(toggle_label, value=st.session_state.dark_mode, key="theme_toggle")
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  HERO — title only, no badge, no subtitle
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
    st.markdown("""
    <div class="banner-warn">
    💡 <b>Demo mode active.</b> Place model files inside a <code>models/</code>
    subfolder next to <code>app.py</code>, then restart.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_vid, col_out = st.columns([1.1, 0.9], gap="large")


# ══════════════════════════════════════════════════════════
#  LEFT — video
# ══════════════════════════════════════════════════════════
with col_vid:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)

    # Card title — live dot shown inline when camera is active, no separate badge pill
 

    # Controls: Start (green) | Stop (red) | Mirror toggle | frame counter
    vc1, vc2, vc3, vc4 = st.columns([1, 1, 1.1, 0.9])
    with vc1:
        st.markdown('<div class="btn-start">', unsafe_allow_html=True)
        if st.button("▶ Start", use_container_width=True,
                     disabled=st.session_state.run_camera, key="btn_start"):
            st.session_state.run_camera = True
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
        st.session_state.mirror = st.toggle("🪞 Mirror", value=st.session_state.mirror)
    with vc4:
        fc = st.session_state.frame_count
        txt = ("🎞 " + str(fc)) if fc else "⏳ Ready"
        st.markdown(
            f'<div style="text-align:center;padding-top:0.45rem;font-size:0.72rem;'
            f'font-weight:700;color:{T["text_muted"]};">{txt}</div>',
            unsafe_allow_html=True,
        )

   

    video_slot = st.empty()
    if not st.session_state.run_camera:
        video_slot.markdown("""
        <div class="video-off">
            <div class="icon">📷</div>
            <div class="msg">Camera is off</div>
            <div class="sub">Press ▶ Start to begin detection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  RIGHT — output
# ══════════════════════════════════════════════════════════
with col_out:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    # st.markdown(
    #     '<div class="card-header"><span class="card-title">📊 PREDICTION OUTPUT</span></div>',
    #     unsafe_allow_html=True,
    # )

    char_slot = st.empty()
    conf_slot = st.empty()

    def render_char_conf(char, conf):
        char_slot.markdown(f"""
        <div class="char-panel">
            <div class="clabel">Detected Character</div>
            <div class="char-big">{char}</div>
        </div>
        """, unsafe_allow_html=True)
        pct = conf * 100
        fill_color = (T["success"] if conf >= 0.80
                      else (T["warning"] if conf >= CONF_THRESH else T["danger"]))
        conf_slot.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-row">
                <span class="conf-row-label">Confidence</span>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{pct:.1f}%;background:{fill_color};"></div>
                </div>
                <span class="conf-pct" style="color:{fill_color};">{pct:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_char_conf(st.session_state.last_char, st.session_state.last_conf)

    word_slot = st.empty()

    def render_word(word):
        display = word if word else "…"
        word_slot.markdown(f"""
        <div class="word-box">
            <div class="wlabel">Formed Word</div>
            <div class="word-text">{display}</div>
        </div>
        """, unsafe_allow_html=True)

    render_word(st.session_state.word)

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
            st.rerun()

   

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA LOOP
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Check permissions and try again.")
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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.frame_count += 1

            char, conf = None, 0.0

            if resource_ok:
                result = hands_det.process(rgb)
                if result.multi_hand_landmarks:
                    hl = result.multi_hand_landmarks[0]
                    roi, bbox = extract_hand_roi(frame, hl)

                    if roi.size > 0:
                        char, conf = predict_character(model, labels, roi)

                    mp_draw.draw_landmarks(
                        rgb,
                        hl,
                        mp_mod.solutions.hands.HAND_CONNECTIONS,
                        mp_mod.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp_mod.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

                    if char and bbox:
                        rgb = draw_overlay(rgb, bbox, char, conf, T)

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


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with 🤟 using
    <strong>Streamlit</strong> ·
    <strong>TensorFlow 2.12</strong> ·
    <strong>MediaPipe</strong> ·
    <strong>OpenCV</strong> ·
    <strong>pyttsx3</strong>
    &nbsp;—&nbsp; Sign Language Translator v3.0
</div>
""", unsafe_allow_html=True)