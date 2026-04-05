

"""
╔══════════════════════════════════════════════════════════════╗
║       Real-Time Sign Language Translator  ·  app.py          ║
║       Premium UI/UX Edition  ·  Dual Theme (Light / Dark)    ║
╚══════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py

Folder structure expected:
    ├── app.py
    └── models/
        ├── mobilenet_mp_25%_v2_best.h5
        └── class_labels_mobilenet_mp_25%_v2.json
"""

# ── Standard library ───────────────────────────────────────────
import os, json, time, threading

# ── Third-party ────────────────────────────────────────────────
import cv2
import numpy as np
import streamlit as st


# ══════════════════════════════════════════════════════════════
#  0 ·  PAGE CONFIG   (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Sign Language Translator",
    page_icon   = "🤟",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)


# ══════════════════════════════════════════════════════════════
#  1 ·  THEME DEFINITIONS
# ══════════════════════════════════════════════════════════════

THEMES = {
    # ── 🌞 Light  ─ warm beige / amber ──────────────────────
    "light": {
        "--bg"           : "#fdf6ec",
        "--bg2"          : "#fff8f0",
        "--surface"      : "#ffffff",
        "--surface2"     : "#fef3e2",
        "--border"       : "#f0d9b5",
        "--accent"       : "#f59e0b",
        "--accent-soft"  : "#fbbf24",
        "--accent-dark"  : "#d97706",
        "--accent-glow"  : "rgba(245,158,11,0.22)",
        "--text"         : "#1c1410",
        "--text-muted"   : "#9a7e5a",
        "--text-inv"     : "#ffffff",
        "--card-shadow"  : "0 4px 32px rgba(245,158,11,0.10)",
        "--live-color"   : "#22c55e",
        "--gradient-hero": "linear-gradient(135deg,#f59e0b 0%,#d97706 100%)",
        "--gradient-word": "linear-gradient(135deg,#f59e0b 0%,#d97706 100%)",
        "--noise-opacity": "0.018",
    },
    # ── 🌙 Dark  ─ deep lavender / violet ───────────────────
    "dark": {
        "--bg"           : "#1a1730",
        "--bg2"          : "#201d3a",
        "--surface"      : "#26234a",
        "--surface2"     : "#302c56",
        "--border"       : "#3d3870",
        "--accent"       : "#a78bfa",
        "--accent-soft"  : "#c4b5fd",
        "--accent-dark"  : "#7c3aed",
        "--accent-glow"  : "rgba(167,139,250,0.25)",
        "--text"         : "#ede9fe",
        "--text-muted"   : "#8b82b5",
        "--text-inv"     : "#ffffff",
        "--card-shadow"  : "0 4px 32px rgba(167,139,250,0.12)",
        "--live-color"   : "#4ade80",
        "--gradient-hero": "linear-gradient(135deg,#7c3aed 0%,#a78bfa 100%)",
        "--gradient-word": "linear-gradient(135deg,#7c3aed 0%,#a78bfa 100%)",
        "--noise-opacity": "0.04",
    },
}


# ══════════════════════════════════════════════════════════════
#  2 ·  SESSION STATE  ─ initialise once
# ══════════════════════════════════════════════════════════════

_defaults = {
    "theme"      : "dark",
    "run_camera" : False,
    "word"       : "",
    "history"    : [],
    "last_char"  : "–",
    "last_conf"  : 0.0,
    "stable_buf" : [],
    "cam_status" : "idle",      # idle | active | error
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════
#  3 ·  CSS INJECTION  ─ theme-aware variables + component styles
# ══════════════════════════════════════════════════════════════

def inject_css(theme: str) -> None:
    t = THEMES[theme]
    # Build CSS variable block from theme dict
    vars_css = "\n    ".join(f"{k}: {v};" for k, v in t.items())

    css = f"""
    <style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

/* ── CSS variables ───────────────────────────────── */
:root {{
    {vars_css}
    --radius-sm : 10px;
    --radius-md : 16px;
    --radius-lg : 24px;
    --radius-xl : 32px;
    --transition: 0.3s cubic-bezier(0.4,0,0.2,1);
}}

/* ── Base reset ──────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    transition: background-color var(--transition), color var(--transition);
}}

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 1.2rem 2.2rem 2.5rem !important;
    max-width: 1440px !important;
}}

/* ═══════════════════════════════════════════════════
   NOISE  TEXTURE OVERLAY  (grain feel)
══════════════════════════════════════════════════════*/
body::before {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    opacity: var(--noise-opacity);
}}

/* ═══════════════════════════════════════════════════
   HEADER BAND
══════════════════════════════════════════════════════*/
.header-band {{
    text-align: center;
    padding: 1.8rem 1rem 0.6rem;
    position: relative;
    z-index: 1;
}}
.header-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 0.28rem 1rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: var(--accent);
    margin-bottom: 0.9rem;
    animation: fadeDown 0.6s ease both;
}}
.header-title {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.9rem, 4vw, 3rem);
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 0.4rem;
    animation: fadeDown 0.7s 0.05s ease both;
}}
.header-title em {{
    font-style: normal;
    color: var(--accent);
}}
.header-sub {{
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 400;
    margin: 0;
    animation: fadeDown 0.7s 0.1s ease both;
}}

/* ═══════════════════════════════════════════════════
   THEME TOGGLE WRAPPER
══════════════════════════════════════════════════════*/
.theme-bar {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 10px;
    padding: 0.3rem 0 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-muted);
}}

/* ═══════════════════════════════════════════════════
   CARDS
══════════════════════════════════════════════════════*/
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--card-shadow);
    padding: 1.5rem;
    transition: box-shadow var(--transition);
    position: relative;
    overflow: hidden;
}}
.card::before {{
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, var(--accent-glow) 0%, transparent 60%);
    opacity: 0;
    transition: opacity var(--transition);
    pointer-events: none;
}}
.card:hover::before {{ opacity: 1; }}

.card-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 7px;
}}

/* ═══════════════════════════════════════════════════
   LIVE DOT
══════════════════════════════════════════════════════*/
.live-dot {{
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--live-color);
    display: inline-block;
    box-shadow: 0 0 0 0 var(--live-color);
    animation: livePulse 1.6s ease-in-out infinite;
}}
@keyframes livePulse {{
    0%   {{ box-shadow: 0 0 0 0   rgba(74,222,128,0.6); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(74,222,128,0);   }}
    100% {{ box-shadow: 0 0 0 0   rgba(74,222,128,0);   }}
}}

/* ═══════════════════════════════════════════════════
   STATUS PILL
══════════════════════════════════════════════════════*/
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 50px;
    padding: 0.3rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.4px;
}}
.status-pill.active  {{ background:#dcfce7; color:#15803d; border:1px solid #86efac; }}
.status-pill.idle    {{ background:var(--surface2); color:var(--text-muted); border:1px solid var(--border); }}
.status-pill.error   {{ background:#fee2e2; color:#dc2626; border:1px solid #fca5a5; }}

/* ── dark-mode overrides for status pills ── */
[data-theme="dark"] .status-pill.active {{ background:#14532d; color:#4ade80; border-color:#166534; }}
[data-theme="dark"] .status-pill.error  {{ background:#450a0a; color:#f87171; border-color:#7f1d1d; }}

/* ═══════════════════════════════════════════════════
   CAMERA PLACEHOLDER
══════════════════════════════════════════════════════*/
.cam-placeholder {{
    background: var(--surface2);
    border: 2px dashed var(--border);
    border-radius: var(--radius-md);
    height: 360px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    color: var(--text-muted);
    transition: border-color var(--transition);
}}
.cam-placeholder .icon {{ font-size: 3.8rem; opacity: 0.6; }}
.cam-placeholder .txt  {{ font-weight: 600; font-size: 0.95rem; }}
.cam-placeholder .hint {{ font-size: 0.8rem; opacity: 0.7; }}

/* ═══════════════════════════════════════════════════
   CHARACTER BOX
══════════════════════════════════════════════════════*/
.char-box {{
    background: var(--surface2);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 0.5rem 0.9rem;
    text-align: center;
    margin-bottom: 0.9rem;
    position: relative;
    overflow: hidden;
    transition: border-color var(--transition);
}}
.char-box:hover {{ border-color: var(--accent-soft); }}
.char-glyph {{
    font-family: 'Syne', sans-serif;
    font-size: 6rem;
    font-weight: 800;
    line-height: 1;
    color: var(--accent);
    display: block;
    filter: drop-shadow(0 0 18px var(--accent-glow));
    animation: charPop 0.18s cubic-bezier(0.34,1.56,0.64,1) both;
}}
@keyframes charPop {{
    0%   {{ transform: scale(0.7); opacity: 0; }}
    100% {{ transform: scale(1);   opacity: 1; }}
}}
.char-label {{
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: var(--text-muted);
    margin-top: 0.35rem;
}}

/* ═══════════════════════════════════════════════════
   CONFIDENCE BAR
══════════════════════════════════════════════════════*/
.conf-wrap {{
    margin-bottom: 0.9rem;
}}
.conf-header {{
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 5px;
}}
.conf-track {{
    height: 7px;
    background: var(--surface2);
    border-radius: 50px;
    border: 1px solid var(--border);
    overflow: hidden;
}}
.conf-fill {{
    height: 100%;
    border-radius: 50px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}}

/* ═══════════════════════════════════════════════════
   WORD BOX
══════════════════════════════════════════════════════*/
.word-box {{
    background: var(--gradient-word);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.1rem 1rem;
    text-align: center;
    margin-bottom: 0.9rem;
    box-shadow: 0 8px 28px var(--accent-glow);
    position: relative;
    overflow: hidden;
}}
.word-box::after {{
    content: '';
    position: absolute;
    top: -40%; right: -20%;
    width: 60%; height: 160%;
    background: rgba(255,255,255,0.07);
    transform: rotate(20deg);
    pointer-events: none;
}}
.word-lbl {{
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.35rem;
}}
.word-text {{
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.08em;
    word-break: break-all;
    min-height: 3.2rem;
    line-height: 1.15;
    text-shadow: 0 2px 12px rgba(0,0,0,0.2);
}}
.word-cursor {{
    display: inline-block;
    width: 3px; height: 2.4rem;
    background: rgba(255,255,255,0.75);
    border-radius: 2px;
    vertical-align: middle;
    margin-left: 3px;
    animation: blink 1s step-start infinite;
}}
@keyframes blink {{
    50% {{ opacity: 0; }}
}}

/* ═══════════════════════════════════════════════════
   ACTION BUTTONS  ─ override Streamlit defaults
══════════════════════════════════════════════════════*/
.stButton > button {{
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 50px !important;
    padding: 0.52rem 1.3rem !important;
    font-size: 0.88rem !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    border: 1.5px solid transparent !important;
    letter-spacing: 0.2px !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 22px var(--accent-glow) !important;
}}
.stButton > button:active {{
    transform: translateY(0px) scale(0.98) !important;
}}

/* ═══════════════════════════════════════════════════
   HISTORY CHIPS
══════════════════════════════════════════════════════*/
.hist-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 0.4rem;
}}
.hist-chip {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 0.22rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent);
    cursor: default;
    transition: background var(--transition);
}}
.hist-chip:hover {{ background: var(--border); }}

/* ═══════════════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════════════════*/
.h-rule {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.9rem 0;
}}

/* ═══════════════════════════════════════════════════
   INFO / WARNING BOX
══════════════════════════════════════════════════════*/
.info-box {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
    line-height: 1.5;
}}

/* ═══════════════════════════════════════════════════
   KEYFRAME HELPERS
══════════════════════════════════════════════════════*/
@keyframes fadeDown {{
    0%   {{ opacity:0; transform:translateY(-14px); }}
    100% {{ opacity:1; transform:translateY(0);     }}
}}
@keyframes fadeIn {{
    from {{ opacity:0; }}
    to   {{ opacity:1; }}
}}

/* ── Streamlit image rounding ──────────────── */
[data-testid="stImage"] img {{
    border-radius: var(--radius-md) !important;
    width: 100% !important;
}}

/* ── Selectbox / toggle ────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stToggle"]    label {{
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  4 ·  RESOURCE LOADING  ─ model + mediapipe (cached)
# ══════════════════════════════════════════════════════════════

MODEL_PATH  = os.path.join("models", "mobilenet_mp_25%_v2_best.h5")
LABELS_PATH = os.path.join("models", "class_labels_mobilenet_mp_25%_v2.json")
IMG_SIZE    = 224
CONF_THRESH = 0.68
STABILITY_N = 8


@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    """Load TF model + reversed label map. Returns (model, labels_list, err)."""
    try:
        import tensorflow as tf
    except ImportError as e:
        return None, None, f"TensorFlow not installed: {e}"

    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model not found: `{MODEL_PATH}`"
    if not os.path.exists(LABELS_PATH):
        return None, None, f"Labels not found: `{LABELS_PATH}`"

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH) as f:
            raw = json.load(f)
        # Support {"A":0,"B":1} → reverse to {0:"A",1:"B"}
        if isinstance(raw, dict):
            if all(isinstance(v, int) for v in raw.values()):
                labels = {v: k for k, v in raw.items()}
                labels = [labels[i] for i in range(len(labels))]
            else:
                labels = [raw[str(i)] for i in range(len(raw))]
        else:
            labels = raw
        return model, labels, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def load_mediapipe():
    """Load MediaPipe Hands. Returns (mp, hands, err)."""
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp.solutions.hands.Hands(
            static_image_mode        = False,
            max_num_hands            = 1,
            min_detection_confidence = 0.65,
            min_tracking_confidence  = 0.55,
        )
        return mp, hands, None
    except ImportError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, str(e)


# ══════════════════════════════════════════════════════════════
#  5 ·  PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════

def extract_hand_roi(frame, hand_landmarks, padding: int = 32):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess(roi, size: int = IMG_SIZE):
    rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rsz  = cv2.resize(rgb, (size, size))
    return np.expand_dims(rsz.astype("float32") / 255.0, axis=0)


def predict_char(model, labels, roi):
    """Return (char, confidence) or (None, 0.0) on failure."""
    if model is None or roi is None or roi.size == 0:
        return None, 0.0
    try:
        preds = model.predict(preprocess(roi), verbose=0)[0]
        idx   = int(np.argmax(preds))
        conf  = float(preds[idx])
        char  = labels[idx] if idx < len(labels) else "?"
        return char, conf
    except Exception:
        return None, 0.0


def draw_overlay(frame_bgr, bbox, char: str, conf: float, theme: str):
    """Draw bounding box + label on frame. Returns modified frame."""
    x1, y1, x2, y2 = bbox
    color = (245, 158, 11) if theme == "light" else (167, 139, 250)  # BGR ≈ amber / violet
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    label = f"{char}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame_bgr, (x1, y1 - th - 14), (x1 + tw + 12, y1), color, -1)
    cv2.putText(frame_bgr, label, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


# ══════════════════════════════════════════════════════════════
#  6 ·  TEXT-TO-SPEECH   (non-blocking)
# ══════════════════════════════════════════════════════════════

def speak(text: str):
    def _worker():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   148)
            engine.setProperty("volume", 0.95)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════
#  7 ·  UI COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════

def conf_bar_html(conf: float) -> str:
    pct = conf * 100
    if conf >= 0.85:
        fill_color = "#22c55e"
        label_color = "#16a34a"
    elif conf >= CONF_THRESH:
        fill_color = "#f59e0b"
        label_color = "#d97706"
    else:
        fill_color = "#ef4444"
        label_color = "#dc2626"
    return f"""
<div class="conf-wrap">
  <div class="conf-header">
    <span>Confidence</span>
    <span style="color:{label_color};font-weight:700;">{pct:.1f}%</span>
  </div>
  <div class="conf-track">
    <div class="conf-fill" style="width:{pct:.1f}%;background:{fill_color};"></div>
  </div>
</div>"""


def char_box_html(char: str) -> str:
    return f"""
<div class="char-box">
  <span class="char-glyph">{char}</span>
  <div class="char-label">Detected Character</div>
</div>"""


def word_box_html(word: str) -> str:
    display = word if word else "…"
    return f"""
<div class="word-box">
  <div class="word-lbl">Formed Word</div>
  <div class="word-text">{display}<span class="word-cursor"></span></div>
</div>"""


def status_pill_html(status: str) -> str:
    cfg = {
        "active" : ("active",  "🟢", "Camera Active"),
        "idle"   : ("idle",    "⚪", "Camera Idle"),
        "error"  : ("error",   "🔴", "Camera Error"),
    }
    cls, icon, label = cfg.get(status, cfg["idle"])
    return f'<span class="status-pill {cls}">{icon} {label}</span>'


def history_chips_html(history: list) -> str:
    if not history:
        return '<p style="color:var(--text-muted);font-size:0.85rem;margin:0;">No words yet.</p>'
    chips = " ".join(
        f'<span class="hist-chip">{w}</span>'
        for w in reversed(history[-12:])
    )
    return f'<div class="hist-wrap">{chips}</div>'


# ══════════════════════════════════════════════════════════════
#  8 ·  MAIN APP  RENDER
# ══════════════════════════════════════════════════════════════

# ── 8a.  Inject CSS for current theme ─────────────────────────
inject_css(st.session_state.theme)

# ── 8b.  Theme toggle (top-right) ─────────────────────────────
tc1, tc2, tc3 = st.columns([6, 1.4, 0.9])
with tc3:
    is_dark = st.session_state.theme == "dark"
    new_dark = st.toggle("🌙", value=is_dark, help="Toggle Light / Dark mode")
    if new_dark != is_dark:
        st.session_state.theme = "dark" if new_dark else "light"
        st.rerun()
with tc2:
    lbl = "Dark Mode" if is_dark else "Light Mode"
    st.markdown(
        f'<div style="text-align:right;padding-top:0.55rem;font-size:0.8rem;'
        f'font-weight:600;color:var(--text-muted);">{lbl}</div>',
        unsafe_allow_html=True,
    )

# ── 8c.  Header ────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
  <div class="header-badge">🤟 AI-Powered Accessibility</div>
  <h1 class="header-title">Real-Time <em>Sign Language</em> Translator</h1>
  <p class="header-sub">Show your hand · Gestures become words · Words become speech</p>
</div>
""", unsafe_allow_html=True)

# ── 8d.  Load resources & surface errors ──────────────────────
model, labels, model_err = load_model_and_labels()
mp_module, hands_det, mp_err = load_mediapipe()
resource_ok = (model is not None) and (hands_det is not None)

if model_err or mp_err:
    ec1, _ = st.columns([3, 1])
    with ec1:
        if model_err:
            st.error(f"⚠️ **Model error:** {model_err}")
        if mp_err:
            st.error(f"⚠️ **MediaPipe error:** {mp_err}")
        st.markdown("""
        <div class="info-box">
        💡 <b>Demo mode active.</b> Place your model files inside a <code>models/</code>
        folder next to <code>app.py</code> and restart.
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  9 ·  MAIN 2-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1.1, 0.9], gap="large")


# ┌─────────────────────────────────────────────────────────────┐
# │  LEFT  ─  Camera                                            │
# └─────────────────────────────────────────────────────────────┘
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ── Card header row ─────────────────────────────────────
    hrow1, hrow2 = st.columns([1, 1])
    with hrow1:
        st.markdown(
            f'<div class="card-label"><span class="live-dot"></span>&nbsp;Live Camera</div>',
            unsafe_allow_html=True,
        )
    with hrow2:
        st.markdown(
            f'<div style="text-align:right;">{status_pill_html(st.session_state.cam_status)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='h-rule'>", unsafe_allow_html=True)

    # ── Control row ─────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1.1])
    with ctrl1:
        start_clicked = st.button(
            "▶ Start", use_container_width=True, type="primary",
            disabled=st.session_state.run_camera,
        )
    with ctrl2:
        stop_clicked = st.button(
            "⏹ Stop", use_container_width=True,
            disabled=not st.session_state.run_camera,
        )
    with ctrl3:
        mirror_on = st.toggle("🪞 Mirror", value=True)

    if start_clicked:
        st.session_state.run_camera  = True
        st.session_state.cam_status  = "active"
        st.toast("Camera started!", icon="📷") 
        st.rerun()
    if stop_clicked:
        st.session_state.run_camera  = False
        st.session_state.cam_status  = "idle"
        st.toast("Camera stopped", icon="⏹")
        st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Video placeholder ────────────────────────────────────
    video_slot = st.empty()

    if not st.session_state.run_camera:
        video_slot.markdown("""
        <div class="cam-placeholder">
          <div class="icon">📷</div>
          <div class="txt">Camera is off</div>
          <div class="hint">Press ▶ Start to begin</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close .card


# ┌─────────────────────────────────────────────────────────────┐
# │  RIGHT  ─  Prediction output                                │
# └─────────────────────────────────────────────────────────────┘
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">📊 &nbsp;Prediction Output</div>', unsafe_allow_html=True)
    st.markdown("<hr class='h-rule'>", unsafe_allow_html=True)

    # Placeholders updated each frame
    char_slot = st.empty()
    conf_slot = st.empty()
    word_slot = st.empty()

    def refresh_output():
        char_slot.markdown(char_box_html(st.session_state.last_char), unsafe_allow_html=True)
        conf_slot.markdown(conf_bar_html(st.session_state.last_conf), unsafe_allow_html=True)
        word_slot.markdown(word_box_html(st.session_state.word),      unsafe_allow_html=True)

    refresh_output()

    st.markdown("<hr class='h-rule'>", unsafe_allow_html=True)

    # ── Action buttons ───────────────────────────────────────
    ab1, ab2, ab3 = st.columns(3)
    with ab1:
        if st.button("🔊 Speak", use_container_width=True, type="primary"):
            if st.session_state.word:
                speak(st.session_state.word)
                # st.toast(f"Speaking: "{st.session_state.word}"", icon="🔊")
                st.toast(f"Speaking: {st.session_state.word}", icon="🔊")
            else:
                st.toast("Nothing to speak yet!", icon="🤫")
    with ab2:
        if st.button("⌫ Delete", use_container_width=True):
            st.session_state.word = st.session_state.word[:-1]
            st.session_state.stable_buf = []
            refresh_output()
    with ab3:
        if st.button("🔄 Reset", use_container_width=True):
            if st.session_state.word:
                st.session_state.history.append(st.session_state.word)
            st.session_state.word        = ""
            st.session_state.stable_buf  = []
            st.session_state.last_char   = "–"
            st.session_state.last_conf   = 0.0
            st.rerun()

    st.markdown("<hr class='h-rule'>", unsafe_allow_html=True)

    # ── Word history ─────────────────────────────────────────
    st.markdown('<div class="card-label">🕘 &nbsp;Word History</div>', unsafe_allow_html=True)
    hist_slot = st.empty()
    hist_slot.markdown(history_chips_html(st.session_state.history), unsafe_allow_html=True)

    if st.session_state.history:
        picked = st.selectbox(
            "Re-speak a past word",
            options=["– select –"] + list(reversed(st.session_state.history[-10:])),
            label_visibility="collapsed",
        )
        if picked != "– select –":
            speak(picked)
            st.toast(f"Speaking: {picked}", icon="🔊")

    st.markdown("</div>", unsafe_allow_html=True)  # close .card


# ══════════════════════════════════════════════════════════════
#  10 ·  CAMERA LOOP
# ══════════════════════════════════════════════════════════════

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.session_state.run_camera = False
        st.session_state.cam_status = "error"
        st.error("❌ Could not open webcam. Check permissions and try again.")
        st.stop()

    try:
        while st.session_state.run_camera:
            st.empty()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.04)
                continue

            if mirror_on:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Hand detection + prediction ──────────────────
            char, conf, bbox = None, 0.0, None
            if resource_ok:
                result = hands_det.process(rgb)
                if result.multi_hand_landmarks:
                    hl = result.multi_hand_landmarks[0]

                    # Draw skeleton on RGB copy
                    mp_module.solutions.drawing_utils.draw_landmarks(
                        rgb, hl,
                        mp_module.solutions.hands.HAND_CONNECTIONS,
                        mp_module.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp_module.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

                    roi, bbox = extract_hand_roi(frame, hl)
                    if roi.size > 0:
                        char, conf = predict_char(model, labels, roi)

                    # Draw bounding box + label (needs BGR then back to RGB)
                    if bbox and char and conf >= CONF_THRESH:
                        tmp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        tmp = draw_overlay(tmp, bbox, char, conf, st.session_state.theme)
                        rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

            # ── Show frame ───────────────────────────────────
            video_slot.image(rgb, channels="RGB", use_container_width=True)

            # ── Stable-character accumulation ────────────────
            if char and conf >= CONF_THRESH:
                st.session_state.last_char = char
                st.session_state.last_conf = conf

                buf = st.session_state.stable_buf
                buf.append(char)
                if len(buf) > STABILITY_N:
                    buf.pop(0)

                # Commit when same char held for STABILITY_N frames
                if len(buf) == STABILITY_N and len(set(buf)) == 1:
                    prev = st.session_state.word
                    if char.lower() == "space":
                        st.session_state.word += " "
                    elif not prev or prev[-1] != char:
                        st.session_state.word += char
                    buf.clear()

            else:
                # Drain buffer when hand lost or low confidence
                if st.session_state.stable_buf:
                    st.session_state.stable_buf.pop(0)
                if not char:
                    st.session_state.last_char = "–"
                    st.session_state.last_conf = 0.0

            # ── Refresh right panel ──────────────────────────
            refresh_output()
            hist_slot.markdown(
                history_chips_html(st.session_state.history), unsafe_allow_html=True
            )

            # ── ~20 fps cap ───────────────────────────────────
            time.sleep(0.05)

    finally:
        cap.release()
        st.session_state.cam_status = "idle"


# ══════════════════════════════════════════════════════════════
#  11 ·  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2.2rem 0 0.6rem;
            color:var(--text-muted);font-size:0.76rem;letter-spacing:0.3px;">
  Built with 🤟 &nbsp;·&nbsp;
  <strong style="color:var(--accent);">Streamlit</strong> ·
  <strong style="color:var(--accent);">TensorFlow 2.12</strong> ·
  <strong style="color:var(--accent);">MediaPipe</strong> ·
  <strong style="color:var(--accent);">OpenCV</strong>
  &nbsp;·&nbsp; Real-Time Sign Language Translator
</div>
""", unsafe_allow_html=True)