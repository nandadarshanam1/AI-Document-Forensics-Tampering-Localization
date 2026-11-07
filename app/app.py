import os
import sys
import re
import subprocess
from pathlib import Path
from textwrap import dedent

import streamlit as st
import pandas as pd

try:
    import fitz
except ImportError:
    fitz = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
FORENSIC_DIR = RESULTS_DIR / "forensic"

FORENSIC_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_SCRIPT = SRC_DIR / "forensic_pipeline.py"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocForensics AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML HELPER
# ============================================================

def html(content):
    st.html(dedent(content))


# ============================================================
# GLOBAL STYLE
# ============================================================

html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  --bg:#09070f;
  --panel:rgba(24,18,34,.72);
  --panel2:rgba(34,24,45,.78);
  --border:rgba(211,184,255,.18);
  --text:#f6f0fb;
  --muted:#b6a9c5;
  --accent:#c4a1ff;
  --accent2:#a78bfa;
  --coral:#ff9b83;
  --green:#72d6ad;
  --yellow:#f3c879;
  --red:#ff7b87;
  --ease:cubic-bezier(.22,1,.36,1);
}

html {scroll-behavior:smooth;}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
body {overflow-x:hidden;}
.stApp {
  color:var(--text);
  background:
    radial-gradient(ellipse at 14% 8%,rgba(133,91,190,.25),transparent 34%),
    radial-gradient(ellipse at 88% 16%,rgba(255,139,117,.15),transparent 30%),
    radial-gradient(ellipse at 55% 92%,rgba(101,69,154,.20),transparent 38%),
    #09070f !important;
  background-attachment:fixed !important;
  position:relative;
}
.stApp:before {
  content:"";position:fixed;inset:-20%;pointer-events:none;z-index:0;
  background:
    radial-gradient(circle at 25% 35%,rgba(172,128,255,.20),transparent 22%),
    radial-gradient(circle at 76% 30%,rgba(255,154,130,.13),transparent 20%),
    radial-gradient(circle at 55% 78%,rgba(114,91,190,.16),transparent 25%);
  filter:blur(48px);animation:chromatic-drift 26s ease-in-out infinite alternate;
}
.stApp:after {
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.13;
  background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
  background-size:54px 54px;
  mask-image:linear-gradient(to bottom,black,transparent 80%);
}
@keyframes chromatic-drift {0%{transform:translate3d(-2%,-1%,0) scale(1)}100%{transform:translate3d(2%,2%,0) scale(1.08)}}
[data-testid="stAppViewContainer"]>.main,[data-testid="stSidebar"],[data-testid="stHeader"],.block-container {position:relative;z-index:1;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stToolbar"] {visibility:hidden;}
[data-testid="stDecoration"] {display:none;}
.block-container {max-width:1500px;padding-top:2rem;padding-bottom:5rem;animation:page-enter .8s var(--ease) both;}
@keyframes page-enter {from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;color:var(--text)!important;letter-spacing:-.025em;}
p,li,label,[data-testid="stMarkdownContainer"] {color:var(--text);}
[data-testid="stCaptionContainer"],small {color:var(--muted)!important;}

/* Translucent sidebar */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,rgba(15,10,23,.94),rgba(22,14,30,.88)) !important;
  border-right:1px solid rgba(211,184,255,.14);
  backdrop-filter:blur(24px) saturate(140%);-webkit-backdrop-filter:blur(24px) saturate(140%);
}
section[data-testid="stSidebar"]>div {padding-top:1.5rem;}
section[data-testid="stSidebar"] * {color:#eee5f8;}

/* Forensic shader atmosphere + cinematic hero */
.stApp {
  background:
    radial-gradient(ellipse at 18% 12%,rgba(125,72,190,.22),transparent 35%),
    radial-gradient(ellipse at 86% 18%,rgba(238,112,132,.13),transparent 32%),
    radial-gradient(ellipse at 52% 92%,rgba(78,55,145,.20),transparent 42%),
    #080710 !important;
}
.stApp:before {
  content:"";position:fixed;inset:-25%;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse at 25% 40%,rgba(139,92,246,.23),transparent 23%),
    radial-gradient(ellipse at 75% 35%,rgba(244,114,182,.15),transparent 20%),
    radial-gradient(ellipse at 52% 78%,rgba(99,102,241,.19),transparent 25%);
  filter:blur(58px);animation:shader-flow 22s ease-in-out infinite alternate;
}
.stApp:after {
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.12;
  background-image:linear-gradient(rgba(210,190,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(210,190,255,.07) 1px,transparent 1px);
  background-size:48px 48px;mask-image:linear-gradient(to bottom,black,transparent 82%);
}
@keyframes shader-flow {0%{transform:translate3d(-3%,-2%,0) scale(.96) rotate(0deg)}50%{transform:translate3d(2%,1%,0) scale(1.06) rotate(3deg)}100%{transform:translate3d(4%,3%,0) scale(1.12) rotate(-2deg)}}
.hero {
  position:relative;isolation:isolate;overflow:hidden;
  display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);
  align-items:center;gap:2rem;
  padding:2.35rem 2.5rem;margin:.15rem 0 1.5rem;border-radius:28px;
  background:linear-gradient(118deg,rgba(20,15,34,.88),rgba(32,20,48,.72) 55%,rgba(42,20,43,.74));
  border:1px solid rgba(221,196,255,.22);
  box-shadow:0 28px 90px rgba(0,0,0,.42),inset 0 1px 0 rgba(255,255,255,.13);
  backdrop-filter:blur(24px) saturate(155%);-webkit-backdrop-filter:blur(24px) saturate(155%);
}
.hero:before {content:"";position:absolute;inset:-70% -20%;z-index:-1;pointer-events:none;background:conic-gradient(from 0deg at 70% 50%,transparent 0deg,rgba(176,129,255,.12) 65deg,transparent 125deg,rgba(255,144,165,.10) 205deg,transparent 280deg);animation:hero-orbit 32s linear infinite;}
.hero:after {content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(105deg,transparent 18%,rgba(255,255,255,.055) 48%,transparent 72%);transform:translateX(-130%);animation:hero-sheen 9s ease-in-out infinite;}
@keyframes hero-orbit {to{transform:rotate(360deg)}}
@keyframes hero-sheen {0%,66%{transform:translateX(-130%)}100%{transform:translateX(130%)}}
.hero-copy{position:relative;z-index:2}
.hero-kicker {display:inline-flex;align-items:center;gap:9px;border:1px solid rgba(216,188,255,.3);border-radius:999px;padding:.42rem .8rem;margin-bottom:1rem;color:#e7d7ff!important;background:rgba(196,161,255,.10);font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;}
.hero h1 {margin:0!important;color:#fff!important;font-size:clamp(2.25rem,4vw,3.65rem)!important;font-weight:800!important;line-height:1.04;letter-spacing:-.055em}
.hero h1 span{background:linear-gradient(100deg,#d7b8ff 0%,#a78bfa 42%,#ff9bb4 90%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hero p {max-width:650px;margin:1rem 0 0!important;color:#c8bdd8!important;font-size:1rem;line-height:1.75}
.hero-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:1.35rem}
.hero-tags span{font-size:.7rem;color:#e9ddf7;border:1px solid rgba(221,196,255,.17);background:rgba(255,255,255,.045);border-radius:999px;padding:6px 10px;letter-spacing:.02em}
.scan-stage{position:relative;min-height:270px;display:flex;align-items:center;justify-content:center;perspective:900px}
.scan-halo{position:absolute;width:250px;height:250px;border-radius:50%;background:radial-gradient(circle,rgba(151,111,255,.25),rgba(236,120,167,.10) 48%,transparent 70%);filter:blur(10px);animation:halo-breathe 5s ease-in-out infinite}
@keyframes halo-breathe{0%,100%{transform:scale(.88);opacity:.65}50%{transform:scale(1.12);opacity:1}}
.scan-orbit{position:absolute;width:255px;height:255px;border:1px solid rgba(203,174,255,.24);border-radius:50%;transform:rotateX(68deg) rotateZ(-24deg);animation:orbit-spin 18s linear infinite}
.scan-orbit:before,.scan-orbit:after{content:"";position:absolute;width:7px;height:7px;border-radius:50%;background:#ff9db5;box-shadow:0 0 18px #ff9db5;top:50%;left:-4px}
.scan-orbit:after{left:auto;right:-4px;background:#bda0ff;box-shadow:0 0 18px #bda0ff}
@keyframes orbit-spin{to{transform:rotateX(68deg) rotateZ(336deg)}}
.doc-card{position:relative;width:178px;height:226px;padding:22px 17px;border-radius:12px;transform:rotateY(-13deg) rotateX(5deg);background:linear-gradient(145deg,rgba(248,243,255,.96),rgba(210,195,231,.88));box-shadow:0 28px 55px rgba(0,0,0,.45),inset 0 1px 0 white;animation:doc-float 6s ease-in-out infinite;overflow:hidden}
@keyframes doc-float{0%,100%{transform:translateY(3px) rotateY(-13deg) rotateX(5deg)}50%{transform:translateY(-10px) rotateY(-5deg) rotateX(2deg)}}
.doc-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:17px;color:#6f5c85;font-size:8px;font-weight:900;letter-spacing:.12em}
.doc-seal{width:21px;height:21px;border:1.5px solid #a78bfa;border-radius:50%;display:grid;place-items:center;color:#8c6bc5;font-size:10px}
.doc-line{height:5px;border-radius:8px;background:rgba(87,68,111,.19);margin:9px 0}
.doc-line.short{width:58%}.doc-line.mid{width:78%}
.doc-highlight{position:absolute;left:14px;top:112px;width:119px;height:30px;border:1.5px solid #ed7099;background:rgba(255,111,157,.17);border-radius:4px;box-shadow:0 0 15px rgba(237,112,153,.35)}
.doc-highlight:after{content:"TAMPER REGION";position:absolute;right:-43px;top:-13px;color:#b34b76;font-size:6px;font-weight:900;letter-spacing:.08em}
.doc-scan{position:absolute;left:0;right:0;height:2px;top:-5px;background:linear-gradient(90deg,transparent,#9d7bff 22%,#ff9bb4 78%,transparent);box-shadow:0 0 16px #b58cff;animation:scan-pass 3.8s ease-in-out infinite}
@keyframes scan-pass{0%{top:0;opacity:0}12%{opacity:1}85%{opacity:1}100%{top:100%;opacity:0}}
.scan-label{position:absolute;bottom:5px;right:5px;color:#75628d;font-size:6px;font-weight:800;letter-spacing:.1em}
.scan-node{position:absolute;width:8px;height:8px;border-radius:50%;background:#b99aff;box-shadow:0 0 18px #b99aff;animation:node-pulse 2s ease-in-out infinite}
.scan-node.n1{top:19%;left:13%}.scan-node.n2{top:25%;right:10%;animation-delay:.6s}.scan-node.n3{bottom:17%;left:18%;animation-delay:1.1s}
@keyframes node-pulse{0%,100%{transform:scale(.7);opacity:.5}50%{transform:scale(1.5);opacity:1}}
@media(max-width:850px){.hero{grid-template-columns:1fr;padding:1.6rem}.scan-stage{min-height:235px}.hero h1{font-size:2.35rem!important}}
/* Glass cards and uploader */
div[data-testid="stFileUploader"],[data-testid="stVerticalBlockBorderWrapper"],div[data-testid="stExpander"],[data-testid="stMetric"] {
  background:linear-gradient(145deg,rgba(39,28,52,.76),rgba(18,13,27,.76)) !important;
  border:1px solid rgba(213,188,244,.18) !important;border-radius:17px;
  box-shadow:0 16px 44px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.055);
  backdrop-filter:blur(18px) saturate(140%);-webkit-backdrop-filter:blur(18px) saturate(140%);
}
div[data-testid="stFileUploader"] {padding:.85rem;transition:transform .35s var(--ease),border-color .35s ease,box-shadow .35s ease;}
div[data-testid="stFileUploader"]:hover {transform:translateY(-2px);border-color:rgba(196,161,255,.42)!important;box-shadow:0 20px 52px rgba(0,0,0,.28);}
[data-testid="stFileUploader"] section {background:rgba(255,255,255,.025);border-radius:12px;}
[data-testid="stFileUploader"] button {border-radius:9px!important;background:rgba(196,161,255,.13)!important;border:1px solid rgba(196,161,255,.3)!important;color:#f5eaff!important;}
[data-testid="stMetric"] {padding:18px;transition:transform .3s var(--ease),border-color .3s ease,box-shadow .3s ease;}
[data-testid="stMetric"]:hover {transform:translateY(-3px);border-color:rgba(255,155,131,.36)!important;box-shadow:0 20px 50px rgba(0,0,0,.28);}
[data-testid="stMetricLabel"] p {color:#b8a9c7!important;font-size:.82rem;font-weight:600;}
[data-testid="stMetricValue"] {color:#fff4ff!important;font-family:'Manrope',sans-serif;font-weight:800;}
[data-testid="stMetricDelta"] {color:var(--green)!important;}

/* Buttons */
.stButton>button,.stDownloadButton>button {
  min-height:2.75rem;border-radius:11px!important;font-weight:700!important;
  border:1px solid rgba(218,192,255,.25)!important;
  background:linear-gradient(120deg,rgba(64,45,82,.88),rgba(39,27,52,.88))!important;
  color:#f6efff!important;transition:all .25s var(--ease);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 8px 24px rgba(0,0,0,.16);
}
.stButton>button:hover,.stDownloadButton>button:hover {transform:translateY(-2px);border-color:rgba(255,163,137,.55)!important;box-shadow:0 12px 30px rgba(151,104,190,.20);}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"] {
  background:linear-gradient(105deg,#9b72e8,#d47d91 58%,#e89a7d)!important;
  border:1px solid rgba(255,220,211,.38)!important;color:#fff!important;
  box-shadow:0 10px 30px rgba(166,103,171,.25),inset 0 1px 0 rgba(255,255,255,.22)!important;
}
.stButton>button[kind="primary"] * {color:white!important;}
.stButton>button:active {transform:translateY(0) scale(.985);}

/* Inputs */
[data-testid="stSelectbox"] div[data-baseweb="select"],[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {
  background:rgba(24,17,34,.88)!important;color:#f6efff!important;border:1px solid rgba(211,184,255,.22)!important;border-radius:10px!important;
}
[data-baseweb="popover"],[data-baseweb="menu"] {background:#21172d!important;}
[data-baseweb="menu"] * {color:#f6efff!important;}
[data-testid="stSelectbox"] label,[data-testid="stTextInput"] label {color:#d3c4e1!important;}

/* Expanders, tabs, alerts, tables */
div[data-testid="stExpander"] {overflow:hidden;transition:transform .3s var(--ease),border-color .3s ease;}
div[data-testid="stExpander"]:hover {transform:translateY(-1px);border-color:rgba(196,161,255,.32)!important;}
div[data-testid="stTabs"] [role="tablist"] {gap:.35rem;border-bottom:1px solid rgba(211,184,255,.16);}
div[data-testid="stTabs"] button[role="tab"] {color:#b9a9c8!important;font-weight:700;border-radius:10px 10px 0 0;}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:#e2ccff!important;border-bottom:3px solid #c4a1ff!important;background:linear-gradient(180deg,rgba(196,161,255,.13),transparent);}
[data-testid="stAlert"] {background:rgba(34,24,45,.82)!important;border:1px solid rgba(211,184,255,.2)!important;border-radius:13px;backdrop-filter:blur(14px);}
[data-testid="stDataFrame"] {border:1px solid rgba(211,184,255,.16);border-radius:13px;overflow:hidden;}
hr {border-color:rgba(211,184,255,.16)!important;}
img {border-radius:13px;}

/* Small interaction details */
.reveal {animation:reveal-up .75s var(--ease) both;}
.delay-1{animation-delay:.1s}.delay-2{animation-delay:.2s}.delay-3{animation-delay:.3s}.delay-4{animation-delay:.4s}
@keyframes reveal-up {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.float-icon {animation:float-icon 4.5s ease-in-out infinite;}
@keyframes float-icon {0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.status-dot {animation:status-pulse 2.2s ease-in-out infinite;}
@keyframes status-pulse {0%,100%{opacity:.65;transform:scale(.9)}50%{opacity:1;transform:scale(1.12)}}
footer {visibility:hidden;}
@media(max-width:800px){.block-container{padding:1rem 1rem 2rem}.hero{padding:1.5rem;border-radius:18px}.hero-title-text{font-size:38px!important}.stApp:before{filter:blur(30px)}.hero:after{display:none}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}}

/* Keep Streamlit's actual application layers above the animated artwork.
   The decorative pseudo-elements must never cover or hide the app widgets. */
.stApp { isolation:isolate !important; }
.stApp:before, .stApp:after { z-index:0 !important; pointer-events:none !important; }
#root, [data-testid="stAppViewContainer"] {
  position:relative !important;
  z-index:1 !important;
  background:transparent !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > section,
[data-testid="stSidebar"],
[data-testid="stHeader"] {
  position:relative !important;
  z-index:2 !important;
}
.main .block-container, .block-container {
  position:relative !important;
  z-index:3 !important;
}
[data-testid="stSidebar"] { z-index:4 !important; }

</style>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html("""
    <div class="reveal" style="
        padding:8px 4px 22px 4px;
        border-bottom:1px solid #3b2b4b;
        margin-bottom:22px;
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:11px;
        ">

            <div class="float-icon" style="
                width:35px;
                height:35px;
                border-radius:10px;

                display:flex;
                align-items:center;
                justify-content:center;

                background:
                    linear-gradient(
                        135deg,
                        #a78bfa,
                        #c4a1ff
                    );

                box-shadow:
                    0 8px 25px
                    rgba(72,217,255,.18);

                font-size:18px;
            ">
                🔎
            </div>

            <div>

                <div style="
                    font-weight:800;
                    font-size:19px;
                    color:#fff7ff;
                ">
                    DocForensics
                </div>

                <div style="
                    color:#c4a1ff;
                    font-size:10px;
                    letter-spacing:1.6px;
                    margin-top:2px;
                ">
                    AI FORENSIC ENGINE
                </div>

            </div>

        </div>

    </div>
    """)

    st.markdown("### System")

    html("""
    <div class="reveal delay-1" style="
        padding:14px;
        border:1px solid #3b2b4b;
        border-radius:13px;
        background:#21172d;
        margin-bottom:12px;
    ">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
        ">

            <span style="
                color:#b6a9c5;
                font-size:13px;
            ">
                Engine
            </span>

            <span style="
                color:#32d583;
                font-size:11px;
                font-weight:700;
            ">
                <span class="status-dot">●</span>
                ONLINE
            </span>

        </div>

        <div style="
            margin-top:10px;
            color:#f6efff;
            font-size:13px;
        ">
            Deep Learning Pipeline
        </div>

    </div>


    <div class="reveal delay-2" style="
        padding:14px;
        border:1px solid #3b2b4b;
        border-radius:13px;
        background:#21172d;
    ">

        <div style="
            color:#b6a9c5;
            font-size:12px;
        ">
            ACTIVE MODELS
        </div>

        <div style="
            margin-top:9px;
            line-height:1.9;
            color:#d8e0e9;
            font-size:12px;
        ">
            <div>◉ ResNet18</div>
            <div>◉ Multimodal Fusion</div>
            <div>◉ U-Net Localization</div>
            <div>◉ Tesseract OCR</div>
        </div>

    </div>
    """)

    st.markdown("---")

    st.markdown("### Supported")

    html("""
    <div class="reveal delay-3" style="
        display:flex;
        flex-wrap:wrap;
        gap:7px;
    ">

        <span style="
            padding:6px 9px;
            border-radius:7px;
            background:#121b28;
            border:1px solid #253246;
            color:#aeb9c7;
            font-size:11px;
        ">PNG</span>

        <span style="
            padding:6px 9px;
            border-radius:7px;
            background:#121b28;
            border:1px solid #253246;
            color:#aeb9c7;
            font-size:11px;
        ">JPG</span>

        <span style="
            padding:6px 9px;
            border-radius:7px;
            background:#121b28;
            border:1px solid #253246;
            color:#aeb9c7;
            font-size:11px;
        ">JPEG</span>

        <span style="
            padding:6px 9px;
            border-radius:7px;
            background:#121b28;
            border:1px solid #253246;
            color:#aeb9c7;
            font-size:11px;
        ">PDF</span>

    </div>
    """)

    st.markdown("---")

    html("""
    <div class="reveal delay-4" style="
        color:#9d8eae;
        font-size:11px;
        line-height:1.6;
    ">

        <b style="color:#8794a6;">
            Research prototype
        </b>

        <br>

        Document tampering detection,
        localization and forensic analysis.

    </div>
    """)


# ============================================================
# HERO
# ============================================================

html("""
<div class="hero">
  <div class="hero-copy">
    <div class="hero-kicker"><span class="status-dot" style="width:7px;height:7px;border-radius:50%;background:#79e0b5;box-shadow:0 0 12px #79e0b5"></span> AI DOCUMENT FORENSICS • LIVE ANALYSIS</div>
    <h1>See beyond the<br><span>surface of a document.</span></h1>
    <p>Multimodal forensic intelligence that examines visual artifacts, OCR inconsistencies and document layout to detect potential manipulation and localize suspicious regions.</p>
    <div class="hero-tags">
      <span>◈ Visual CNN</span><span>⌗ OCR inspection</span><span>▧ Tamper localization</span><span>✧ Explainable assessment</span>
    </div>
  </div>
  <div class="scan-stage" aria-label="Animated document forensic scan illustration">
    <div class="scan-halo"></div><div class="scan-orbit"></div>
    <div class="scan-node n1"></div><div class="scan-node n2"></div><div class="scan-node n3"></div>
    <div class="doc-card">
      <div class="doc-scan"></div>
      <div class="doc-top"><span>FORENSIC / 001</span><span class="doc-seal">⌕</span></div>
      <div class="doc-line mid"></div><div class="doc-line"></div><div class="doc-line short"></div>
      <div class="doc-line mid" style="margin-top:24px"></div><div class="doc-line"></div>
      <div class="doc-highlight"></div>
      <div class="doc-line short" style="margin-top:31px"></div><div class="doc-line mid"></div>
      <div class="scan-label">PIXEL + TEXT ANALYSIS</div>
    </div>
  </div>
</div>
""")


# ============================================================
# UPLOAD SECTION
# ============================================================

html("""
<div class="reveal" style="
    margin-bottom:10px;
    color:#f6f0fb;
    font-size:18px;
    font-weight:700;
">
    Analyze a document
</div>

<div class="reveal delay-1" style="
    margin-bottom:14px;
    color:#aa9ab9;
    font-size:13px;
">
    Upload an invoice, certificate, form, receipt
    or other document.
</div>
""")

uploaded = st.file_uploader(
    "Upload document",
    type=["png", "jpg", "jpeg", "pdf"],
    label_visibility="collapsed",
)


# ============================================================
# FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    safe_name = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        Path(uploaded_file.name).stem
    )

    output = (
        FORENSIC_DIR /
        f"uploaded_{safe_name}{suffix}"
    )

    with open(output, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    return output


def render_pdf(pdf_path):

    if fitz is None:

        raise RuntimeError(
            "PyMuPDF is not installed.\n\n"
            "Run:\n"
            "pip install pymupdf"
        )

    document = fitz.open(
        str(pdf_path)
    )

    pages = []

    pdf_dir = (
        FORENSIC_DIR /
        "pdf_pages"
    )

    pdf_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, page in enumerate(
        document
    ):

        matrix = fitz.Matrix(
            2.0,
            2.0
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        page_path = (
            pdf_dir /
            f"{pdf_path.stem}_page_{index + 1}.png"
        )

        pix.save(
            str(page_path)
        )

        pages.append(
            page_path
        )

    document.close()

    return pages


def run_forensic_pipeline(image_path):

    if not PIPELINE_SCRIPT.exists():

        raise FileNotFoundError(
            "Forensic pipeline not found:\n"
            f"{PIPELINE_SCRIPT}"
        )

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        str(image_path),
    ]

    try:

        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )

    except subprocess.TimeoutExpired:

        raise RuntimeError(
            "Forensic pipeline timed out "
            "after 10 minutes."
        )

    if result.returncode != 0:

        stderr_text = result.stderr.strip()
        stdout_text = result.stdout.strip()

        details = []
        if stderr_text:
            details.append("STDERR:\\n" + stderr_text)
        if stdout_text:
            details.append("STDOUT:\\n" + stdout_text)
        if not details:
            details.append(
                "The pipeline exited with a non-zero status and returned no output."
            )

        raise RuntimeError(
            f"Forensic pipeline failed (exit code {result.returncode}).\\n\\n"
            + "\\n\\n".join(details)
        )

    return result.stdout


def parse_number(
    text,
    pattern,
    default=0.0
):

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if not match:
        return default

    try:
        return float(
            match.group(1)
        )

    except Exception:
        return default


def parse_pipeline_output(
    output,
    image_path
):

    visual = parse_number(
        output,
        r"Visual tampering probability:\s*([0-9.]+)%"
    )

    multimodal = parse_number(
        output,
        r"Multimodal tampering probability:\s*([0-9.]+)%"
    )

    combined = parse_number(
        output,
        r"Combined tampering risk:\s*([0-9.]+)%"
    )

    suspicious_area = parse_number(
        output,
        r"Suspicious pixel area:\s*([0-9.]+)%"
    )

    regions = parse_number(
        output,
        r"Suspicious regions:\s*([0-9]+)"
    )

    words = parse_number(
        output,
        r"OCR words detected:\s*([0-9]+)"
    )

    ocr_conf = parse_number(
        output,
        r"OCR average confidence:\s*([0-9.]+)%"
    )

    if combined >= 70:

        verdict = "HIGH TAMPERING RISK"
        level = "HIGH"

    elif combined >= 40:

        verdict = "MODERATE TAMPERING RISK"
        level = "MODERATE"

    else:

        verdict = "LOW TAMPERING RISK"
        level = "LOW"

    stem = image_path.stem

    predicted_mask = (
        FORENSIC_DIR /
        f"{stem}_predicted_mask.png"
    )

    highlighted = (
        FORENSIC_DIR /
        f"{stem}_highlighted.png"
    )

    ocr_csv = (
        FORENSIC_DIR /
        f"{stem}_ocr.csv"
    )

    report = (
        FORENSIC_DIR /
        f"{stem}_report.txt"
    )

    return {

        "visual": visual,

        "multimodal": multimodal,

        "combined": combined,

        "suspicious_area":
            suspicious_area,

        "regions":
            int(regions),

        "words":
            int(words),

        "ocr_conf":
            ocr_conf,

        "verdict":
            verdict,

        "level":
            level,

        "predicted_mask":
            predicted_mask,

        "highlighted":
            highlighted,

        "ocr_csv":
            ocr_csv,

        "report":
            report,

        "raw_output":
            output,
    }


def analyze_single_image(
    image_path
):

    output = run_forensic_pipeline(
        image_path
    )

    return parse_pipeline_output(
        output,
        image_path
    )


def load_ocr_csv(path):

    if path is None:
        return None

    if not Path(path).exists():
        return None

    try:

        df = pd.read_csv(path)

        return df

    except Exception:

        return None


def suspicious_ocr(df):

    if df is None or df.empty:
        return pd.DataFrame()

    possible_columns = [
        "tampering_overlap",
        "overlap",
        "risk",
    ]

    selected = None

    for col in possible_columns:

        if col in df.columns:

            selected = col
            break

    if selected is None:
        return pd.DataFrame()

    try:

        values = pd.to_numeric(
            df[selected],
            errors="coerce"
        ).fillna(0)

        return df[
            values > 0.15
        ].copy()

    except Exception:

        return pd.DataFrame()


def image_exists(path):

    return (
        path is not None
        and Path(path).exists()
    )


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded is None:

    html("""
    <div class="reveal" style="
        margin-top:25px;

        padding:
            55px 30px;

        border:
            1px dashed #2b3b52;

        border-radius:
            20px;

        text-align:
            center;

        background:
            radial-gradient(
                circle at center,
                rgba(77,124,255,.08),
                transparent 55%
            ),
            #181222;

        transition:
            transform .4s var(--ease),
            box-shadow .4s ease;
    ">

        <div class="float-icon" style="
            font-size:42px;
            margin-bottom:15px;
        ">
            📄
        </div>

        <div style="
            font-size:19px;
            font-weight:700;
            color:#f6efff;
        ">
            Drop a document above
        </div>

        <div style="
            color:#aa9ab9;
            font-size:13px;
            margin-top:8px;
        ">
            PDF or image
            •
            AI forensic analysis
            •
            OCR
            •
            localization
        </div>

    </div>
    """)

    st.stop()


# ============================================================
# SAVE UPLOAD
# ============================================================

try:

    uploaded_path = (
        save_uploaded_file(
            uploaded
        )
    )

except Exception as exc:

    st.error(
        f"Could not save uploaded file: {exc}"
    )

    st.stop()


# ============================================================
# PREVIEW
# ============================================================

extension = (
    uploaded_path.suffix.lower()
)


if extension == ".pdf":

    try:

        page_paths = render_pdf(
            uploaded_path
        )

    except Exception as exc:

        st.error(str(exc))
        st.stop()

    if not page_paths:

        st.error(
            "The PDF contains no readable pages."
        )

        st.stop()

    html("""
    <div class="reveal" style="
        margin-top:30px;
        margin-bottom:15px;
    ">

        <div style="
            color:#f6f0fb;
            font-size:20px;
            font-weight:700;
        ">
            PDF document
        </div>

        <div style="
            color:#aa9ab9;
            font-size:12px;
            margin-top:4px;
        ">
            Multi-page forensic analysis
        </div>

    </div>
    """)

    page_number = st.selectbox(
        "Preview page",
        range(
            1,
            len(page_paths) + 1
        ),
        format_func=lambda x:
            f"Page {x}",
    )

    selected_page = (
        page_paths[
            page_number - 1
        ]
    )

    st.image(
        str(selected_page),
        caption=f"Page {page_number}",
        width="stretch",
    )

    pages_to_process = (
        page_paths
    )

else:

    page_paths = [
        uploaded_path
    ]

    # Images have a single page to analyze.
    pages_to_process = [
        uploaded_path
    ]

    html("""
    <div class="reveal" style="
        margin-top:30px;
        margin-bottom:15px;

        color:#f6f0fb;
        font-size:20px;
        font-weight:700;
    ">
        Document preview
    </div>
    """)

    st.image(
        str(uploaded_path),
        width="stretch",
    )


# ============================================================
# RUN ANALYSIS BUTTON
# ============================================================

st.markdown("")

analyze = st.button(
    "🔎  Run AI Forensic Analysis",
    type="primary",
    use_container_width=True,
)

if not analyze:
    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

all_results = []

progress = st.progress(0)

status = st.empty()

for index, page_path in enumerate(
    pages_to_process
):

    status.info(
        f"Analyzing page "
        f"{index + 1} of "
        f"{len(pages_to_process)}..."
    )

    try:

        result = (
            analyze_single_image(
                page_path
            )
        )

        all_results.append(
            result
        )

    except Exception as exc:

        status.error(
            f"Page {index + 1} failed: "
            f"{exc}"
        )

        all_results.append({

            "visual": 0,

            "multimodal": 0,

            "combined": 0,

            "suspicious_area": 0,

            "regions": 0,

            "words": 0,

            "ocr_conf": 0,

            "verdict":
                "ANALYSIS FAILED",

            "level":
                "ERROR",

            "predicted_mask":
                None,

            "highlighted":
                None,

            "ocr_csv":
                None,

            "report":
                None,

            "raw_output":
                str(exc),
        })

    progress.progress(
        (index + 1) /
        len(pages_to_process)
    )


status.success(
    "Forensic analysis complete."
)

progress.empty()


# ============================================================
# VALID RESULTS
# ============================================================

valid_results = [
    x
    for x in all_results
    if x["level"] != "ERROR"
]

if not valid_results:

    st.error(
        "No page could be analyzed. The forensic pipeline failed for every page."
    )

    st.markdown("### Pipeline error details")
    st.caption(
        "These details identify the actual failure. Expand the page below and "
        "copy the error text if you need further debugging."
    )

    for error_index, error_result in enumerate(all_results):
        with st.expander(
            f"Page {error_index + 1} • Analysis error",
            expanded=True,
        ):
            st.code(
                error_result.get("raw_output", "No error output was captured."),
                language="text",
            )

    st.info(
        "Check that src\\forensic_pipeline.py exists and that the project's "
        "Python environment contains all required packages and model files."
    )
    st.stop()


# ============================================================
# DOCUMENT AGGREGATION
# ============================================================

document_visual = max(
    x["visual"]
    for x in valid_results
)

document_multimodal = max(
    x["multimodal"]
    for x in valid_results
)

document_risk = max(
    x["combined"]
    for x in valid_results
)

document_area = max(
    x["suspicious_area"]
    for x in valid_results
)

document_regions = sum(
    x["regions"]
    for x in valid_results
)

document_words = sum(
    x["words"]
    for x in valid_results
)

ocr_values = [
    x["ocr_conf"]
    for x in valid_results
    if x["ocr_conf"] > 0
]

document_ocr_conf = (

    sum(ocr_values) /
    len(ocr_values)

    if ocr_values

    else 0
)


# ============================================================
# VERDICT
# ============================================================

if document_risk >= 70:

    document_verdict = (
        "HIGH TAMPERING RISK"
    )

    verdict_color = "#ff5d73"

    verdict_bg = (
        "rgba(255,93,115,.08)"
    )

elif document_risk >= 40:

    document_verdict = (
        "MODERATE TAMPERING RISK"
    )

    verdict_color = "#f5c451"

    verdict_bg = (
        "rgba(245,196,81,.08)"
    )

else:

    document_verdict = (
        "LOW TAMPERING RISK"
    )

    verdict_color = "#32d583"

    verdict_bg = (
        "rgba(50,213,131,.08)"
    )


# ============================================================
# RESULT HEADER
# ============================================================

html(f"""
<div class="reveal" style="
    margin-top:45px;
    margin-bottom:22px;
">

    <div style="
        color:#6f7d91;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.6px;
        text-transform:uppercase;
    ">
        FORENSIC RESULT
    </div>

    <div style="
        margin-top:7px;
        color:#fff7ff;
        font-size:29px;
        font-weight:800;
        letter-spacing:-.7px;
    ">
        Analysis complete
    </div>

</div>


<div class="reveal delay-1" style="
    padding:24px;

    border-radius:18px;

    border:
        1px solid
        {verdict_color}33;

    background:
        {verdict_bg};

    margin-bottom:25px;
">

    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:20px;
        flex-wrap:wrap;
    ">

        <div>

            <div style="
                color:#8491a4;
                font-size:11px;
                letter-spacing:1.5px;
                font-weight:700;
            ">
                DOCUMENT VERDICT
            </div>

            <div style="
                margin-top:7px;
                color:{verdict_color};
                font-size:25px;
                font-weight:800;
            ">
                {document_verdict}
            </div>

            <div style="
                margin-top:6px;
                color:#8996a8;
                font-size:12px;
            ">
                Based on visual, multimodal
                and localization evidence.
            </div>

        </div>


        <div style="
            min-width:130px;
            text-align:right;
        ">

            <div style="
                color:#758398;
                font-size:11px;
                letter-spacing:1px;
            ">
                COMBINED RISK
            </div>

            <div style="
                margin-top:3px;
                color:#fff7ff;
                font-size:38px;
                font-weight:800;
            ">
                {document_risk:.2f}%
            </div>

        </div>

    </div>

</div>
""")


# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4, m5 = st.columns(5)

with m1:

    st.metric(
        "Visual Risk",
        f"{document_visual:.2f}%"
    )

with m2:

    st.metric(
        "Multimodal Risk",
        f"{document_multimodal:.2f}%"
    )

with m3:

    st.metric(
        "Suspicious Area",
        f"{document_area:.2f}%"
    )

with m4:

    st.metric(
        "Suspicious Regions",
        document_regions
    )

with m5:

    st.metric(
        "OCR Words",
        document_words
    )


# ============================================================
# PAGE RESULTS
# ============================================================

if len(valid_results) > 1:

    html("""
    <div class="reveal" style="
        margin-top:42px;
        margin-bottom:14px;
        font-size:19px;
        font-weight:700;
        color:#f6f0fb;
    ">
        Page-by-page findings
    </div>
    """)

    for index, result in enumerate(
        all_results
    ):

        if result["level"] == "ERROR":
            continue

        page_label = (
            f"Page {index + 1}"
        )

        with st.expander(
            f"{page_label}  •  "
            f"{result['combined']:.2f}% risk  •  "
            f"{result['verdict']}",
            expanded=(index == 0),
        ):

            a, b, c, d = st.columns(4)

            with a:

                st.metric(
                    "Visual",
                    f"{result['visual']:.2f}%"
                )

            with b:

                st.metric(
                    "Multimodal",
                    f"{result['multimodal']:.2f}%"
                )

            with c:

                st.metric(
                    "Suspicious Area",
                    f"{result['suspicious_area']:.2f}%"
                )

            with d:

                st.metric(
                    "OCR Confidence",
                    f"{result['ocr_conf']:.2f}%"
                )

            if image_exists(
                result["highlighted"]
            ):

                st.image(
                    str(
                        result["highlighted"]
                    ),
                    caption=(
                        "Detected suspicious regions"
                    ),
                    width="stretch",
                )


# ============================================================
# FORENSIC EVIDENCE
# ============================================================

html("""
<div class="reveal" style="
    margin-top:42px;
    margin-bottom:17px;
    color:#f6f0fb;
    font-size:21px;
    font-weight:750;
">
    Forensic evidence
</div>
""")


best_index = max(
    range(len(all_results)),
    key=lambda i:
        all_results[i]["combined"]
)

best_result = (
    all_results[best_index]
)


left, right = st.columns(2)


with left:

    html("""
    <div class="reveal delay-1" style="
        margin-bottom:12px;
        color:#a5b0bf;
        font-size:13px;
        font-weight:600;
    ">
        Suspicious-region localization
    </div>
    """)

    if image_exists(
        best_result["highlighted"]
    ):

        st.image(
            str(
                best_result["highlighted"]
            ),
            width="stretch",
        )

    else:

        st.info(
            "Highlighted forensic image "
            "was not generated."
        )


with right:

    html("""
    <div class="reveal delay-2" style="
        margin-bottom:12px;
        color:#a5b0bf;
        font-size:13px;
        font-weight:600;
    ">
        Predicted tampering mask
    </div>
    """)

    if image_exists(
        best_result["predicted_mask"]
    ):

        st.image(
            str(
                best_result["predicted_mask"]
            ),
            width="stretch",
        )

    else:

        st.info(
            "Predicted localization mask "
            "was not generated."
        )


# ============================================================
# MODEL EVIDENCE
# ============================================================

html("""
<div class="reveal" style="
    margin-top:42px;
    margin-bottom:15px;
    color:#f6f0fb;
    font-size:21px;
    font-weight:750;
">
    Model evidence
</div>
""")


e1, e2, e3 = st.columns(3)


with e1:

    html(f"""
    <div class="reveal delay-1" style="
        padding:20px;
        border:1px solid #3b2b4b;
        border-radius:15px;
        background:#21172d;
        height:100%;

        transition:
            transform .4s var(--ease),
            box-shadow .4s ease;
    ">

        <div style="
            color:#a99ab9;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            VISUAL MODEL
        </div>

        <div style="
            margin-top:10px;
            color:#f6efff;
            font-size:18px;
            font-weight:700;
        ">
            ResNet18
        </div>

        <div style="
            margin-top:8px;
            color:#b6a9c5;
            font-size:12px;
            line-height:1.6;
        ">
            Image-only document tampering
            classifier.
        </div>

        <div style="
            margin-top:16px;
            color:#c4a1ff;
            font-size:24px;
            font-weight:800;
        ">
            {document_visual:.2f}%
        </div>

        <div style="
            color:#9d8eae;
            font-size:11px;
        ">
            tampering probability
        </div>

    </div>
    """)


with e2:

    html(f"""
    <div class="reveal delay-2" style="
        padding:20px;
        border:1px solid #3b2b4b;
        border-radius:15px;
        background:#21172d;
        height:100%;

        transition:
            transform .4s var(--ease),
            box-shadow .4s ease;
    ">

        <div style="
            color:#a99ab9;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            MULTIMODAL MODEL
        </div>

        <div style="
            margin-top:10px;
            color:#f6efff;
            font-size:18px;
            font-weight:700;
        ">
            Visual + OCR + Layout
        </div>

        <div style="
            margin-top:8px;
            color:#b6a9c5;
            font-size:12px;
            line-height:1.6;
        ">
            Fuses visual document features
            with textual and layout evidence.
        </div>

        <div style="
            margin-top:16px;
            color:#a78bfa;
            font-size:24px;
            font-weight:800;
        ">
            {document_multimodal:.2f}%
        </div>

        <div style="
            color:#9d8eae;
            font-size:11px;
        ">
            tampering probability
        </div>

    </div>
    """)


with e3:

    html(f"""
    <div class="reveal delay-3" style="
        padding:20px;
        border:1px solid #3b2b4b;
        border-radius:15px;
        background:#21172d;
        height:100%;

        transition:
            transform .4s var(--ease),
            box-shadow .4s ease;
    ">

        <div style="
            color:#a99ab9;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            LOCALIZATION MODEL
        </div>

        <div style="
            margin-top:10px;
            color:#f6efff;
            font-size:18px;
            font-weight:700;
        ">
            U-Net v2
        </div>

        <div style="
            margin-top:8px;
            color:#b6a9c5;
            font-size:12px;
            line-height:1.6;
        ">
            Pixel-level localization of
            suspicious document regions.
        </div>

        <div style="
            margin-top:16px;
            color:#32d583;
            font-size:24px;
            font-weight:800;
        ">
            {document_regions}
        </div>

        <div style="
            color:#9d8eae;
            font-size:11px;
        ">
            suspicious regions detected
        </div>

    </div>
    """)


# ============================================================
# OCR FORENSICS
# ============================================================

html("""
<div class="reveal" style="
    margin-top:42px;
    margin-bottom:15px;
    color:#f6f0fb;
    font-size:21px;
    font-weight:750;
">
    OCR forensics
</div>
""")


o1, o2, o3 = st.columns(3)


with o1:

    st.metric(
        "OCR words",
        document_words
    )


with o2:

    st.metric(
        "Average confidence",
        f"{document_ocr_conf:.2f}%"
    )


with o3:

    st.metric(
        "Pages analyzed",
        len(all_results)
    )


best_ocr_df = load_ocr_csv(
    best_result["ocr_csv"]
)


if (
    best_ocr_df is not None
    and not best_ocr_df.empty
):

    with st.expander(
        "View extracted OCR data",
        expanded=False
    ):

        st.dataframe(
            best_ocr_df,
            use_container_width=True,
            hide_index=True,
        )


    suspicious_df = (
        suspicious_ocr(
            best_ocr_df
        )
    )


    if not suspicious_df.empty:

        html("""
        <div class="reveal" style="
            margin-top:17px;
            margin-bottom:9px;
            color:#ffcf68;
            font-size:13px;
            font-weight:700;
        ">
            OCR regions overlapping suspicious pixels
        </div>
        """)

        st.dataframe(
            suspicious_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# REPORT
# ============================================================

html("""
<div class="reveal" style="
    margin-top:42px;
    margin-bottom:15px;
    color:#f6f0fb;
    font-size:21px;
    font-weight:750;
">
    Forensic report
</div>
""")


report_parts = []


for index, result in enumerate(
    all_results
):

    report_path = (
        result.get("report")
    )

    if (
        report_path
        and Path(report_path).exists()
    ):

        try:

            text = Path(
                report_path
            ).read_text(
                encoding="utf-8",
                errors="replace"
            )

            report_parts.append(

                f"\n{'=' * 70}\n"
                f"PAGE {index + 1}\n"
                f"{'=' * 70}\n\n"
                f"{text}"

            )

        except Exception:

            pass


combined_report = "".join(
    report_parts
)


if not combined_report:

    combined_report = (

        "DocForensics AI Report\n\n"

        f"Verdict: "
        f"{document_verdict}\n"

        f"Combined Risk: "
        f"{document_risk:.2f}%\n"

        f"Visual Risk: "
        f"{document_visual:.2f}%\n"

        f"Multimodal Risk: "
        f"{document_multimodal:.2f}%\n"

        f"Suspicious Area: "
        f"{document_area:.2f}%\n"

        f"Suspicious Regions: "
        f"{document_regions}\n"

        f"OCR Words: "
        f"{document_words}\n"

        f"OCR Confidence: "
        f"{document_ocr_conf:.2f}%\n"
    )


st.download_button(

    label=
        "⬇️  Download forensic report",

    data=
        combined_report,

    file_name=
        "docforensics_report.txt",

    mime=
        "text/plain",

    use_container_width=True,
)


# ============================================================
# RAW OUTPUT
# ============================================================

with st.expander(
    "Developer / model output",
    expanded=False
):

    for index, result in enumerate(
        all_results
    ):

        st.markdown(
            f"### Page {index + 1}"
        )

        st.code(
            result.get(
                "raw_output",
                ""
            ),
            language="text",
        )


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="reveal" style="
    margin-top:55px;

    padding-top:20px;

    border-top:
        1px solid #3b2b4b;

    text-align:center;

    color:#566478;

    font-size:11px;

    line-height:1.7;
">

    <b style="color:#758398;">
        DocForensics AI
    </b>

    &nbsp;•&nbsp;

    Deep Learning Document Forensics

    &nbsp;•&nbsp;

    Research Prototype

    <br>

    Results indicate potential manipulation risk
    and should not be treated as legal authentication.

</div>
""")