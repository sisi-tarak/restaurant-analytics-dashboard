"""
Restaurant Analytics Dashboard  v3.0
======================================
Tech Stack : Streamlit · Pandas · NumPy · Plotly
Dataset    : Restaurant Dataset

Features
---------
- Light / Dark theme toggle
- 5-tab layout: Overview · Advanced · Rankings · Insights · Map & Data
- 20+ interactive Plotly charts
- Treemap, Sunburst, Parallel Coordinates, Box plots, Violin
- Underrated gems finder, Restaurant comparison tool
- Percentile ranking table
- Sentiment analysis, Price elasticity
- Geo map with color-coded markers
- Full upload support for any CSV (up to 750 MB)
- session_state–based upload (file_id key — zero re-reads on reruns)
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# UPLOAD LIMIT — set programmatically as a safety net.
# The CLI flag  --server.maxUploadSize=1024  in run.bat is the primary method.
# config.toml is the secondary method.
# This block is a third fallback that works when neither of the above loads.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from streamlit import config as _stcfg
    _stcfg.set_option("server.maxUploadSize", 1024)
except Exception:
    pass  # server options can only be set before the server starts; ignore safely

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍽 Restaurant Analytics Dashboard",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ──────────────────────────────────────────────────────────────────────────────
DARK = {
    "bg":        "#0b0b0f",
    "surface":   "#13131a",
    "border":    "#1e1e2e",
    "text":      "#f1f5f9",
    "muted":     "#94a3b8",
    "inputbg":   "#0b0b0f",
    "tag":       "dark",
}
LIGHT = {
    "bg":        "#f4f6fb",
    "surface":   "#ffffff",
    "border":    "#dde3ef",
    "text":      "#0f172a",
    "muted":     "#64748b",
    "inputbg":   "#ffffff",
    "tag":       "light",
}

# Accent palette — identical in both themes
C_GREEN  = "#00c27a"
C_BLUE   = "#3b82f6"
C_PURPLE = "#8b5cf6"
C_AMBER  = "#f59e0b"
C_RED    = "#ef4444"
C_CYAN   = "#06b6d4"
C_PINK   = "#ec4899"
PALETTE  = [C_BLUE, C_GREEN, C_PURPLE, C_AMBER, C_RED,
            C_CYAN, C_PINK, "#84cc16", "#fb923c", "#a78bfa"]

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "dark_mode":       True,
        "dataset_choice":  None,   # None=landing, "default"=use bundled, "upload"=custom
        "file_id":         None,
        "uploaded_df":     None,
        "upload_error":    None,
        "col_info":        None,   # dict describing detected columns
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ──────────────────────────────────────────────────────────────────────────────
# ACTIVE THEME HELPER
# ──────────────────────────────────────────────────────────────────────────────
def T() -> dict:
    return DARK if st.session_state.dark_mode else LIGHT

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS  = ["Name", "City", "Cuisine", "Rating", "Votes", "Cost"]
OPTIONAL_COLS  = ["Location", "Locality", "Online Delivery",
                  "Table Booking", "Restaurant Type", "Latitude", "Longitude"]
SYNTHETIC_COLS = ["Online Delivery", "Table Booking",
                  "Restaurant Type", "Latitude", "Longitude"]

def detect_columns(raw_df: pd.DataFrame) -> dict:
    """
    Given the RAW dataframe (before preprocessing), returns a dict describing
    which columns were found, which will be synthesized, and which are missing.
    """
    raw_cols_lower = {c.strip().lower() for c in raw_df.columns}

    # alias map mirrors _preprocess
    alias = {
        "restaurant name": "Name", "name": "Name",
        "city": "City",
        "cuisine": "Cuisine", "cuisines": "Cuisine",
        "aggregate rating": "Rating", "rating": "Rating",
        "votes": "Votes",
        "average cost for two": "Cost",
        "approx_cost(for two people)": "Cost",
        "cost for two": "Cost",
        "cost": "Cost",
        "location": "Location", "locality": "Locality",
        "online_order": "Online Delivery",
        "book_table": "Table Booking",
        "listed_in(type)": "Restaurant Type",
        "latitude": "Latitude", "longitude": "Longitude",
    }
    resolved = {alias[k] for k in alias if k in raw_cols_lower}

    found   = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in resolved]
    missing = [c for c in REQUIRED_COLS if c not in resolved]
    synth   = [c for c in SYNTHETIC_COLS if c not in resolved]

    return {
        "found":   found,
        "missing": missing,
        "synth":   synth,
        "total_rows": len(raw_df),
        "total_cols": len(raw_df.columns),
        "raw_names":  list(raw_df.columns),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ──────────────────────────────────────────────────────────────────────────────
def landing_page():
    """Welcome screen shown before any dataset is chosen."""
    # ── Hide sidebar + collapse control completely on landing page ──────────
    st.markdown("""
<style>
  [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"],
  section[data-testid="stSidebar"] {
      display: none !important;
      width: 0 !important;
      min-width: 0 !important;
      max-width: 0 !important;
      overflow: hidden !important;
  }
  /* Expand main area to full width when sidebar is hidden */
  [data-testid="stMain"],
  .main .block-container {
      margin-left: 0 !important;
      padding-left: 2rem !important;
  }
</style>
""", unsafe_allow_html=True)
    inject_css()
    t = T()

    # ── Theme toggle top-right ─────────────────────────────────────────────
    _, tr = st.columns([8, 1])
    with tr:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(icon, help="Toggle Light / Dark theme", key="lp_theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # ── Hero ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; padding: 48px 0 32px;">
        <div style="font-size:3rem; margin-bottom:10px;">🍽</div>
        <div style="font-size:2rem; font-weight:800; color:{t['text']};
                    letter-spacing:-0.5px; margin-bottom:8px;">
            Restaurant Analytics Dashboard
        </div>
        <div style="font-size:0.9rem; color:{t['muted']}; max-width:480px;
                    margin:0 auto;">
            Interactive data visualization powered by Streamlit and Plotly.
            Choose how you'd like to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two option cards ───────────────────────────────────────────────────
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"""
        <div class="landing-card">
            <div class="landing-card-icon">📦</div>
            <div class="landing-card-title">Use Default Dataset</div>
            <div class="landing-card-desc">
                Start instantly with the bundled restaurant dataset.<br>
                6,593 restaurants · 23 cities · India
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        if st.button("Continue with Default Dataset",
                     use_container_width=True, type="primary", key="btn_default"):
            st.session_state.dataset_choice = "default"
            st.rerun()

    with right:
        st.markdown(f"""
        <div class="landing-card">
            <div class="landing-card-icon">📂</div>
            <div class="landing-card-title">Upload Your Own Dataset</div>
            <div class="landing-card-desc">
                Upload any CSV file.<br>
                Required columns: Name, City, Cuisine, Rating, Votes, Cost
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        if st.button("Upload a New Dataset",
                     use_container_width=True, key="btn_upload"):
            st.session_state.dataset_choice = "upload"
            st.rerun()

    # ── If upload mode is chosen, show uploader right on this page ─────────
    if st.session_state.dataset_choice == "upload":
        st.markdown(f"""
        <div style="margin-top:28px; padding:24px; background:{t['surface']};
                    border:1px solid {t['border']}; border-radius:12px;">
            <div style="font-size:0.85rem; font-weight:700; color:{t['text']};
                        margin-bottom:12px;">Upload your CSV file</div>
        """, unsafe_allow_html=True)

        up = st.file_uploader(
            "Drag & drop or browse  (max 1 GB, CSV only)",
            type=["csv"], key="landing_uploader",
            help="Required: Name, City, Cuisine, Rating, Votes, Cost",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if up is not None:
            size_mb = up.size / (1024 * 1024)
            if size_mb > 1024:
                st.error(f"File too large ({size_mb:.1f} MB). Max is 1 GB.", icon="🚫")
            else:
                file_id = getattr(up, "file_id", None) or f"{up.name}_{up.size}"
                if file_id != st.session_state.file_id:
                    with st.spinner(f"Processing {up.name} …"):
                        try:
                            raw_bytes = up.read()
                            raw_df_preview = pd.read_csv(
                                io.BytesIO(raw_bytes), nrows=5, low_memory=False)
                            st.session_state.col_info   = detect_columns(
                                pd.read_csv(io.BytesIO(raw_bytes), low_memory=False))
                            st.session_state.uploaded_df = load_uploaded(raw_bytes)
                            st.session_state.file_id     = file_id
                            st.session_state.upload_error = None
                            del raw_bytes
                        except Exception as e:
                            st.session_state.upload_error = str(e)
                            st.session_state.uploaded_df  = None

                if st.session_state.upload_error:
                    st.error(f"Could not read file: {st.session_state.upload_error}", icon="🚫")
                elif st.session_state.uploaded_df is not None:
                    _show_col_info(up.name, size_mb)
                    st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
                    if st.button("Open Dashboard →",
                                 use_container_width=True, type="primary",
                                 key="btn_open_dash"):
                        st.rerun()

        # Back button
        st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
        if st.button("← Back", key="btn_back"):
            st.session_state.dataset_choice = None
            st.rerun()

    st.markdown(f"""
    <div style="text-align:center; color:{t['muted']}; font-size:0.72rem;
                margin-top:48px; padding-top:16px;
                border-top:1px solid {t['border']};">
        Restaurant Analytics Dashboard &nbsp;·&nbsp;
        Streamlit &nbsp;·&nbsp; Plotly
    </div>""", unsafe_allow_html=True)


def _show_col_info(filename: str, size_mb: float):
    """Render the column detection summary card after a successful upload."""
    t   = T()
    info = st.session_state.col_info
    if not info:
        return

    found_html = "".join(f'<span class="col-badge col-found">{c}</span>'
                         for c in info["found"])
    synth_html = "".join(f'<span class="col-badge col-synth">{c} (auto-generated)</span>'
                         for c in info["synth"])
    miss_html  = "".join(f'<span class="col-badge col-missing">{c} MISSING</span>'
                         for c in info["missing"])

    synth_row  = f'<div style="margin-bottom:4px;">{synth_html}</div>' if synth_html else ""
    miss_row   = f'<div style="margin-bottom:4px;">{miss_html}</div>'  if miss_html  else ""
    if info["missing"]:
        status_row = (f'<div style="font-size:0.72rem;color:{C_RED};margin-top:6px;">'
                      f'⚠ Missing required columns — some charts may not render.</div>')
    else:
        status_row = (f'<div style="font-size:0.72rem;color:{C_GREEN};margin-top:6px;">'
                      f'✓ All required columns found.</div>')

    st.markdown(f"""<div style="background:{t['surface']};border:1px solid {t['border']};
border-left:3px solid {C_GREEN};border-radius:0 10px 10px 0;
padding:16px 18px;margin-top:12px;">
<div style="font-size:0.75rem;font-weight:700;color:{t['text']};margin-bottom:8px;">
{filename} &nbsp;·&nbsp; {size_mb:.1f} MB &nbsp;·&nbsp; {info['total_rows']:,} rows &nbsp;·&nbsp; {info['total_cols']} columns detected
</div>
<div style="margin-bottom:6px;">{found_html}</div>
{synth_row}
{miss_row}
{status_row}
</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (injected after theme is known)
# ──────────────────────────────────────────────────────────────────────────────
def inject_css():
    t = T()
    st.markdown(f"""
<style>
  /* ── Base ──────────────────────────────────────────────────────────────── */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] {{
      background-color: {t['bg']} !important;
      color: {t['text']} !important;
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }}
  [data-testid="stHeader"] {{ background: transparent !important; }}

  /* ── Sidebar ───────────────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {{
      background-color: {t['surface']} !important;
      border-right: 1px solid {t['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stTextInput > div > div input {{
      background-color: {t['inputbg']} !important;
      border: 1px solid {t['border']} !important;
      border-radius: 6px !important;
      color: {t['text']} !important;
  }}
  [data-testid="stSidebar"] label {{
      font-size: 0.75rem !important;
      font-weight: 600 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.06em !important;
      color: {t['muted']} !important;
  }}

  /* ── Tabs ──────────────────────────────────────────────────────────────── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
      background: {t['surface']};
      border-bottom: 2px solid {t['border']};
      gap: 4px;
  }}
  [data-testid="stTabs"] [data-baseweb="tab"] {{
      background: transparent !important;
      color: {t['muted']} !important;
      font-weight: 600;
      font-size: 0.82rem;
      border-radius: 6px 6px 0 0;
      padding: 10px 18px;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
      background: {t['bg']} !important;
      color: {C_BLUE} !important;
      border-bottom: 2px solid {C_BLUE} !important;
  }}
  [data-testid="stTabContent"] {{
      background: {t['bg']};
      padding: 12px 0;
  }}

  /* ── KPI Cards ─────────────────────────────────────────────────────────── */
  .kpi-wrap {{
      background: {t['surface']};
      border: 1px solid {t['border']};
      border-radius: 10px;
      padding: 18px 20px 14px;
      position: relative;
      overflow: hidden;
  }}
  .kpi-wrap::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
  }}
  .kpi-blue::before   {{ background: {C_BLUE};   }}
  .kpi-green::before  {{ background: {C_GREEN};  }}
  .kpi-purple::before {{ background: {C_PURPLE}; }}
  .kpi-amber::before  {{ background: {C_AMBER};  }}
  .kpi-cyan::before   {{ background: {C_CYAN};   }}
  .kpi-red::before    {{ background: {C_RED};    }}

  .kpi-label {{
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: {t['muted']};
      margin-bottom: 8px;
  }}
  .kpi-value {{
      font-size: 1.85rem;
      font-weight: 700;
      line-height: 1;
      color: {t['text']};
      letter-spacing: -0.5px;
  }}
  .kpi-sub {{
      font-size: 0.68rem;
      color: {t['muted']};
      margin-top: 5px;
  }}
  .kpi-icon {{
      float: right;
      font-size: 1.1rem;
      opacity: 0.4;
      margin-top: -22px;
  }}

  /* ── Section label ─────────────────────────────────────────────────────── */
  .sec-title {{
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: {t['muted']};
      border-bottom: 1px solid {t['border']};
      padding-bottom: 7px;
      margin: 24px 0 14px;
  }}

  /* ── Page header ───────────────────────────────────────────────────────── */
  .page-title {{
      font-size: 1.75rem;
      font-weight: 800;
      color: {t['text']};
      letter-spacing: -0.5px;
  }}
  .page-sub {{
      font-size: 0.8rem;
      color: {t['muted']};
      margin-top: 3px;
  }}

  /* ── Chart card ────────────────────────────────────────────────────────── */
  .chart-card {{
      background: {t['surface']};
      border: 1px solid {t['border']};
      border-radius: 10px;
      padding: 4px 8px 2px;
      margin-bottom: 14px;
  }}

  /* ── Insight box ───────────────────────────────────────────────────────── */
  .insight-box {{
      background: {t['surface']};
      border-left: 3px solid {C_BLUE};
      border-radius: 0 8px 8px 0;
      padding: 10px 14px;
      margin: 6px 0 14px;
      font-size: 0.82rem;
      color: {t['muted']};
  }}

  /* ── Expander ──────────────────────────────────────────────────────────── */
  [data-testid="stExpander"] {{
      background: {t['surface']} !important;
      border: 1px solid {t['border']} !important;
      border-radius: 10px !important;
  }}
  [data-testid="stExpander"] summary {{
      color: {t['text']} !important;
      font-weight: 600 !important;
  }}

  /* ── Buttons — all variants, both themes ────────────────────────────────── */

  /* PRIMARY buttons — maximum specificity to beat Streamlit defaults */
  button[data-testid="baseButton-primary"],
  [data-testid="baseButton-primary"],
  .stButton > button[kind="primary"],
  .stButton button[kind="primary"],
  div[data-testid="column"] .stButton > button[kind="primary"] {{
      background-color: {C_BLUE} !important;
      background: {C_BLUE} !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 6px !important;
      font-weight: 600 !important;
  }}
  button[data-testid="baseButton-primary"]:hover,
  [data-testid="baseButton-primary"]:hover,
  .stButton > button[kind="primary"]:hover {{
      background-color: #2563eb !important;
      background: #2563eb !important;
      color: #ffffff !important;
  }}
  /* Force white text on ALL child nodes inside primary button */
  button[data-testid="baseButton-primary"] *,
  [data-testid="baseButton-primary"] *,
  .stButton > button[kind="primary"] *,
  .stButton > button[kind="primary"] p,
  .stButton > button[kind="primary"] span,
  .stButton > button[kind="primary"] div {{
      color: #ffffff !important;
  }}

  /* Secondary / default buttons */
  [data-testid="baseButton-secondary"],
  .stButton > button:not([kind="primary"]) {{
      background-color: {t['surface']} !important;
      color: {t['text']} !important;
      border: 1px solid {t['border']} !important;
      border-radius: 6px !important;
      font-weight: 600 !important;
  }}
  [data-testid="baseButton-secondary"]:hover,
  .stButton > button:not([kind="primary"]):hover {{
      background-color: {t['border']} !important;
      color: {t['text']} !important;
  }}

  /* Catch-all: every button's <p> or <span> text node */
  .stButton button p,
  .stButton button span,
  [data-testid="baseButton-primary"] p,
  [data-testid="baseButton-primary"] span,
  [data-testid="baseButton-secondary"] p,
  [data-testid="baseButton-secondary"] span {{
      color: inherit !important;
  }}

  /* Download button */
  .stDownloadButton button {{
      background: {C_BLUE} !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 6px !important;
      font-weight: 600 !important;
  }}
  .stDownloadButton button p,
  .stDownloadButton button span {{
      color: #ffffff !important;
  }}

  /* ── Upload widget — full override for both themes ─────────────────────── */
  [data-testid="stFileUploader"],
  [data-testid="stFileUploader"] > div,
  [data-testid="stFileUploader"] section {{
      background-color: {t['inputbg']} !important;
      border-radius: 8px !important;
  }}
  [data-testid="stFileUploader"] section {{
      border: 1.5px dashed {t['border']} !important;
      padding: 16px !important;
  }}
  /* Drop-zone text and icon */
  [data-testid="stFileUploader"] section span,
  [data-testid="stFileUploader"] section p,
  [data-testid="stFileUploader"] section small {{
      color: {t['muted']} !important;
  }}
  /* Browse button inside uploader */
  [data-testid="stFileUploader"] section button {{
      background-color: {t['surface']} !important;
      color: {t['text']} !important;
      border: 1px solid {t['border']} !important;
      border-radius: 6px !important;
  }}
  /* Uploaded file name row */
  [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
      background-color: {t['surface']} !important;
      border: 1px solid {t['border']} !important;
      border-radius: 6px !important;
      color: {t['text']} !important;
  }}
  [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {{
      color: {t['text']} !important;
  }}

  /* ── Landing page cards ─────────────────────────────────────────────────── */
  .landing-card {{
      background: {t['surface']};
      border: 1.5px solid {t['border']};
      border-radius: 14px;
      padding: 32px 28px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, box-shadow 0.2s;
  }}
  .landing-card:hover {{
      border-color: {C_BLUE};
      box-shadow: 0 0 0 3px rgba(59,130,246,0.12);
  }}
  .landing-card-icon {{
      font-size: 2.4rem;
      margin-bottom: 12px;
  }}
  .landing-card-title {{
      font-size: 1.05rem;
      font-weight: 700;
      color: {t['text']};
      margin-bottom: 6px;
  }}
  .landing-card-desc {{
      font-size: 0.78rem;
      color: {t['muted']};
      line-height: 1.5;
  }}

  /* ── Column info badges ─────────────────────────────────────────────────── */
  .col-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.7rem;
      font-weight: 600;
      margin: 3px;
  }}
  .col-found   {{ background: rgba(0,194,122,0.15); color: {C_GREEN}; border: 1px solid {C_GREEN}; }}
  .col-synth   {{ background: rgba(245,158,11,0.12); color: {C_AMBER}; border: 1px solid {C_AMBER}; }}
  .col-missing {{ background: rgba(239,68,68,0.12); color: {C_RED}; border: 1px solid {C_RED}; }}

  /* ── Footer ────────────────────────────────────────────────────────────── */
  .footer-bar {{
      text-align: center;
      color: {t['muted']};
      font-size: 0.7rem;
      letter-spacing: 0.04em;
      padding: 20px 0 8px;
      border-top: 1px solid {t['border']};
      margin-top: 28px;
  }}

  /* ── Scrollbar ─────────────────────────────────────────────────────────── */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {t['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CITY COORDS
# ──────────────────────────────────────────────────────────────────────────────
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),     "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867), "Ahmedabad": (23.0225, 72.5714),
    "Chennai": (13.0827, 80.2707),   "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),    "Chandigarh": (30.7333, 76.7794),
    "Surat": (21.1702, 72.8311),     "Lucknow": (26.8467, 80.9462),
    "Kochi": (9.9312, 76.2673),      "Indore": (22.7196, 75.8577),
    "Gurgaon": (28.4595, 77.0266),   "Noida": (28.5355, 77.3910),
    "Bhopal": (23.2599, 77.4126),    "Nagpur": (21.1458, 79.0882),
    "Patna": (25.5941, 85.1376),     "Coimbatore": (11.0168, 76.9558),
    "Vadodara": (22.3072, 73.1812),  "Agra": (27.1767, 78.0081),
    "Nashik": (19.9975, 73.7898),
}


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    alias = {
        "restaurant name": "Name", "name": "Name",
        "city": "City",
        "cuisine": "Cuisine", "cuisines": "Cuisine",
        "aggregate rating": "Rating", "rating": "Rating",
        "votes": "Votes",
        "average cost for two": "Cost",
        "approx_cost(for two people)": "Cost",
        "cost for two": "Cost",
        "cost": "Cost",
        "location": "Location", "locality": "Locality",
        "online_order": "Online Delivery",
        "book_table": "Table Booking",
        "listed_in(type)": "Restaurant Type",
        "latitude": "Latitude", "longitude": "Longitude",
    }
    lc = {c.lower(): c for c in df.columns}
    df.rename(columns={lc[k]: v for k, v in alias.items()
                       if k in lc and lc[k] != v}, inplace=True)

    for col in ["Name", "City", "Cuisine", "Rating", "Votes", "Cost"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df[df["Rating"].notna() & (df["Rating"] > 0)].copy()
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
    df["Cost"] = pd.to_numeric(
        df["Cost"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce")
    df["Cost"] = df["Cost"].fillna(df["Cost"].median())

    if "Online Delivery" not in df.columns:
        df["Online Delivery"] = np.random.default_rng(42).choice(
            ["Yes", "No"], len(df), p=[0.55, 0.45])
    else:
        df["Online Delivery"] = (df["Online Delivery"].astype(str).str.strip()
                                  .replace({"1":"Yes","0":"No","True":"Yes","False":"No"}))

    if "Table Booking" not in df.columns:
        df["Table Booking"] = np.random.default_rng(7).choice(
            ["Yes", "No"], len(df), p=[0.35, 0.65])
    else:
        df["Table Booking"] = (df["Table Booking"].astype(str).str.strip()
                                .replace({"1":"Yes","0":"No","True":"Yes","False":"No"}))

    if "Restaurant Type" not in df.columns:
        def _type(c):
            c = str(c).lower()
            if any(x in c for x in ["cafe","coffee","bakery"]):      return "Café"
            if any(x in c for x in ["fast food","burger","pizza"]):  return "Quick Bites"
            if any(x in c for x in ["dessert","ice cream","shake"]): return "Dessert Shop"
            if any(x in c for x in ["bar","pub","brewery"]):         return "Bar & Lounge"
            if any(x in c for x in ["buffet","thali"]):              return "Buffet"
            return "Casual Dining"
        df["Restaurant Type"] = df["Cuisine"].apply(_type)

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        rng = np.random.default_rng(13)
        def _coords(row):
            base = CITY_COORDS.get(row["City"], (20.5937, 78.9629))
            return base[0]+rng.uniform(-0.25,0.25), base[1]+rng.uniform(-0.25,0.25)
        coords = df.apply(_coords, axis=1)
        df["Latitude"]  = coords.apply(lambda x: x[0])
        df["Longitude"] = coords.apply(lambda x: x[1])

    def _band(r):
        if r>=4.5: return "Excellent  4.5+"
        if r>=4.0: return "Very Good  4.0–4.4"
        if r>=3.5: return "Good  3.5–3.9"
        if r>=3.0: return "Average  3.0–3.4"
        return "Below Avg  <3.0"

    def _tier(c):
        if   c<=300:  return "Budget  ≤₹300"
        elif c<=700:  return "Affordable  ₹301–700"
        elif c<=1200: return "Mid-Range  ₹701–1200"
        elif c<=2000: return "Premium  ₹1201–2000"
        else:         return "Luxury  ₹2000+"

    df["Rating Band"]     = df["Rating"].apply(_band)
    df["Price Tier"]      = df["Cost"].apply(_tier)
    df["Primary Cuisine"] = df["Cuisine"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown")
    df["Rating Pct"] = df["Rating"].rank(pct=True).mul(100).round(1)

    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_default(path: str) -> pd.DataFrame:
    return _preprocess(pd.read_csv(path, low_memory=False))


def load_uploaded(file_bytes: bytes) -> pd.DataFrame:
    """Load from raw bytes — NOT cached (already in session_state)."""
    buf = io.BytesIO(file_bytes)
    chunks = pd.read_csv(buf, chunksize=50_000, low_memory=False)
    return _preprocess(pd.concat(chunks, ignore_index=True))


# ──────────────────────────────────────────────────────────────────────────────
# FILTER
# ──────────────────────────────────────────────────────────────────────────────
def apply_filters(df, search, city, cuisine, r_type,
                  rating_rng, cost_rng, min_votes, online_opt, table_opt):
    f = df.copy()
    if search:
        f = f[f["Name"].str.contains(search, case=False, na=False)]
    if city != "All Cities":
        f = f[f["City"] == city]
    if cuisine != "All Cuisines":
        f = f[f["Cuisine"].str.contains(cuisine, case=False, na=False)]
    if r_type != "All Types":
        f = f[f["Restaurant Type"] == r_type]
    f = f[(f["Rating"] >= rating_rng[0]) & (f["Rating"] <= rating_rng[1])]
    f = f[(f["Cost"]   >= cost_rng[0])   & (f["Cost"]   <= cost_rng[1])]
    f = f[f["Votes"] >= min_votes]
    if online_opt != "All": f = f[f["Online Delivery"] == online_opt]
    if table_opt  != "All": f = f[f["Table Booking"]   == table_opt]
    return f.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def sidebar(df: pd.DataFrame):
    t = T()
    with st.sidebar:
        # ── Brand + theme toggle ───────────────────────────────────────────
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
            <div style="padding:14px 0 4px">
                <div style="font-size:1.1rem;font-weight:800;color:{t['text']};
                            letter-spacing:-0.3px">Restaurant Analytics</div>
                <div style="font-size:0.7rem;color:{t['muted']};margin-top:2px">
                    Interactive Dashboard
                </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div style='padding-top:18px'/>", unsafe_allow_html=True)
            icon = "☀️" if st.session_state.dark_mode else "🌙"
            if st.button(icon, help="Toggle Light / Dark theme",
                         use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

        st.markdown(f"<hr style='border:none;border-top:1px solid {t['border']};margin:10px 0 16px'/>",
                    unsafe_allow_html=True)

        # ── Upload ────────────────────────────────────────────────────────
        st.markdown(f"<div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.1em;color:{t['muted']};margin-bottom:6px'>"
                    f"Dataset</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload CSV  (max 1 GB)",
            type=["csv"],
            help="Required columns: Name, City, Cuisine, Rating, Votes, Cost. Max size: 1 GB.",
        )

        # ── KEY FIX: use file_id, never call .read() here ─────────────────
        if uploaded is not None:
            size_mb = uploaded.size / (1024 * 1024)
            if size_mb > 1024:
                st.error(f"File too large ({size_mb:.1f} MB). Max is 1 GB.", icon="🚫")
                uploaded = None
            else:
                st.success(f"{uploaded.name}  ·  {size_mb:.1f} MB", icon="📂")

        st.markdown(f"<hr style='border:none;border-top:1px solid {t['border']};margin:14px 0'/>",
                    unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────────────────────────
        st.markdown(f"<div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.1em;color:{t['muted']};margin-bottom:10px'>"
                    f"Filters</div>", unsafe_allow_html=True)

        search = st.text_input("Search restaurant name", placeholder="e.g. McDonald's")

        cities = ["All Cities"] + sorted(df["City"].dropna().unique().tolist())
        cuisines = ["All Cuisines"] + sorted({
            c.strip() for row in df["Cuisine"].dropna()
            for c in row.split(",") if c.strip()})
        types = ["All Types"] + sorted(df["Restaurant Type"].dropna().unique().tolist())

        city    = st.selectbox("City",            cities)
        cuisine = st.selectbox("Cuisine",         cuisines)
        r_type  = st.selectbox("Restaurant Type", types)

        r_lo, r_hi = float(df["Rating"].min()), float(df["Rating"].max())
        rating_rng = st.slider("Rating", r_lo, r_hi, (r_lo, r_hi), 0.1)

        c_lo, c_hi = int(df["Cost"].min()), int(df["Cost"].max())
        cost_rng   = st.slider("Cost for Two (₹)", c_lo, c_hi, (c_lo, c_hi), 100)

        v_lo = int(df["Votes"].min())
        v_hi = int(df["Votes"].max())
        min_votes = st.slider("Min. Votes", v_lo, v_hi, v_lo, 5)

        online_opt = st.radio("Online Delivery", ["All","Yes","No"], horizontal=True)
        table_opt  = st.radio("Table Booking",   ["All","Yes","No"], horizontal=True)

        st.markdown(f"<hr style='border:none;border-top:1px solid {t['border']};margin:14px 0'/>",
                    unsafe_allow_html=True)

        # ── Back to Home ──────────────────────────────────────────────────
        if st.button("← Change Dataset / Home", use_container_width=True,
                     key="sb_back_home"):
            st.session_state.dataset_choice = None
            st.session_state.file_id        = None
            st.session_state.uploaded_df    = None
            st.session_state.upload_error   = None
            st.session_state.col_info       = None
            st.rerun()

        st.caption("Restaurant Dataset")

    return (uploaded, search, city, cuisine, r_type,
            rating_rng, cost_rng, min_votes, online_opt, table_opt)


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT HELPER
# ──────────────────────────────────────────────────────────────────────────────
def _lay(**kw):
    t = T()
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["muted"], family="Inter, Segoe UI, sans-serif", size=11),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=t["border"],
                    borderwidth=1, font=dict(color=t["muted"], size=10)),
        xaxis=dict(gridcolor=t["border"], zerolinecolor=t["border"],
                   tickfont=dict(color=t["muted"]), linecolor=t["border"]),
        yaxis=dict(gridcolor=t["border"], zerolinecolor=t["border"],
                   tickfont=dict(color=t["muted"]), linecolor=t["border"]),
        title_font=dict(color=t["text"], size=13,
                        family="Inter, Segoe UI, sans-serif"),
        colorway=PALETTE,
    )
    base.update(kw)
    return base

def W(fig):
    fig.update_layout(**_lay())
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────
def kpi_cards(fdf):
    n      = len(fdf)
    avg_r  = fdf["Rating"].mean() if n else 0
    top_c  = fdf["Primary Cuisine"].value_counts().index[0] if n else "—"
    avg_co = fdf["Cost"].mean() if n else 0
    pct_od = (fdf["Online Delivery"]=="Yes").mean()*100 if n else 0
    pct_tb = (fdf["Table Booking"]  =="Yes").mean()*100 if n else 0

    cards = [
        ("Total Restaurants", f"{n:,}",            "Matching filters",          "kpi-blue",   "$"),
        ("Avg Rating",        f"{avg_r:.2f} / 5",  "Across filtered set",       "kpi-green",  "★"),
        ("Top Cuisine",       top_c[:16],           "Most represented",          "kpi-purple", "◈"),
        ("Avg Cost for Two",  f"₹{avg_co:,.0f}",   "Per filtered selection",    "kpi-amber",  "₹"),
        ("Online Delivery",   f"{pct_od:.1f}%",    "Offer delivery",            "kpi-cyan",   "↑"),
        ("Table Booking",     f"{pct_tb:.1f}%",    "Accept reservations",       "kpi-red",    "□"),
    ]
    cols = st.columns(6)
    for col, (label, value, sub, cls, icon) in zip(cols, cards):
        col.markdown(f"""
        <div class="kpi-wrap {cls}">
            <div class="kpi-label">{label}<span class="kpi-icon">{icon}</span></div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── TAB 1: OVERVIEW CHARTS ───────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def ch_city_bar(fdf):
    d = fdf["City"].value_counts().nlargest(15).reset_index()
    d.columns = ["City","Count"]
    fig = px.bar(d, x="City", y="Count", color="Count",
                 color_continuous_scale=[C_BLUE, C_GREEN],
                 title="Restaurants by City")
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return W(fig)

def ch_cuisine_donut(fdf):
    d = fdf["Primary Cuisine"].value_counts().nlargest(10).reset_index()
    d.columns = ["Cuisine","Count"]
    fig = px.pie(d, names="Cuisine", values="Count",
                 title="Cuisine Share  (Top 10)",
                 hole=0.52, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+percent",
                      textfont=dict(size=10), pull=[0.02]*len(d))
    return W(fig)

def ch_rating_hist(fdf):
    fig = px.histogram(fdf, x="Rating", nbins=20,
                       color_discrete_sequence=[C_BLUE],
                       marginal="box", title="Rating Distribution")
    fig.update_traces(marker_line_width=0, opacity=0.85)
    return W(fig)

def ch_price_bar(fdf):
    ORDER = ["Budget  ≤₹300","Affordable  ₹301–700","Mid-Range  ₹701–1200",
             "Premium  ₹1201–2000","Luxury  ₹2000+"]
    d = fdf["Price Tier"].value_counts().reindex(ORDER, fill_value=0).reset_index()
    d.columns = ["Tier","Count"]
    fig = px.bar(d, x="Tier", y="Count", color="Tier",
                 color_discrete_sequence=[C_GREEN,C_BLUE,C_PURPLE,C_AMBER,C_RED],
                 title="Price Range Distribution")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(showlegend=False)
    return W(fig)

def ch_scatter(fdf):
    s = fdf.sample(min(1200, len(fdf)), random_state=42)
    fig = px.scatter(s, x="Cost", y="Rating",
                     color="Primary Cuisine", size="Votes", size_max=22,
                     opacity=0.72, hover_name="Name",
                     hover_data=["City","Votes"],
                     title="Cost vs Rating  (bubble = votes)",
                     color_discrete_sequence=PALETTE)
    return W(fig)

def ch_city_cost(fdf):
    d = (fdf.groupby("City")["Cost"].mean()
         .sort_values(ascending=False).nlargest(15).reset_index())
    d.columns = ["City","Avg Cost"]
    fig = px.bar(d, x="Avg Cost", y="City", orientation="h",
                 color="Avg Cost",
                 color_continuous_scale=[C_BLUE,C_AMBER],
                 title="Avg Cost for Two by City", text="Avg Cost")
    fig.update_traces(texttemplate="₹%{text:.0f}", textposition="outside",
                      marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return W(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ── TAB 2: ADVANCED ANALYTICS ────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def ch_treemap(fdf):
    d = (fdf.groupby(["City","Primary Cuisine"])
         .agg(count=("Name","count"), avg_rating=("Rating","mean"))
         .reset_index())
    fig = px.treemap(d, path=["City","Primary Cuisine"],
                     values="count", color="avg_rating",
                     color_continuous_scale=[C_RED,C_AMBER,C_GREEN],
                     title="Treemap  —  City › Cuisine  (colour = avg rating)")
    fig.update_traces(textinfo="label+value")
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=T()["muted"])))
    return W(fig)

def ch_sunburst(fdf):
    d = (fdf.groupby(["City","Restaurant Type","Primary Cuisine"])
         .size().reset_index(name="Count")
         .query("Count >= 3"))
    fig = px.sunburst(d, path=["City","Restaurant Type","Primary Cuisine"],
                      values="Count",
                      color="Count",
                      color_continuous_scale=[C_BLUE,C_PURPLE,C_PINK],
                      title="Sunburst  —  City › Type › Cuisine")
    fig.update_coloraxes(showscale=False)
    return W(fig)

def ch_boxplot(fdf):
    fig = px.box(fdf, x="Restaurant Type", y="Rating",
                 color="Restaurant Type",
                 color_discrete_sequence=PALETTE,
                 points="outliers",
                 title="Rating Distribution by Restaurant Type")
    fig.update_layout(showlegend=False)
    return W(fig)

def ch_violin(fdf):
    top_cities = fdf["City"].value_counts().nlargest(6).index
    sub = fdf[fdf["City"].isin(top_cities)]
    fig = px.violin(sub, x="City", y="Rating",
                    color="City", box=True,
                    color_discrete_sequence=PALETTE,
                    title="Rating Spread by City  (violin + box)")
    fig.update_layout(showlegend=False)
    return W(fig)

def ch_parallel(fdf):
    s = fdf.sample(min(1500, len(fdf)), random_state=1)
    fig = px.parallel_coordinates(
        s,
        dimensions=["Rating","Votes","Cost"],
        color="Rating",
        color_continuous_scale=[C_RED,C_AMBER,C_GREEN],
        title="Parallel Coordinates  —  Rating · Votes · Cost",
    )
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=T()["muted"])))
    return W(fig)

def ch_heatmap(fdf):
    top_c  = fdf["Primary Cuisine"].value_counts().nlargest(10).index
    top_ct = fdf["City"].value_counts().nlargest(10).index
    sub    = fdf[fdf["Primary Cuisine"].isin(top_c) & fdf["City"].isin(top_ct)]
    pivot  = sub.pivot_table(index="Primary Cuisine", columns="City",
                              values="Rating", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,C_RED],[0.5,C_AMBER],[1,C_GREEN]],
        text=np.round(pivot.values,2), texttemplate="%{text}",
        colorbar=dict(tickfont=dict(color=T()["muted"]),outlinecolor=T()["border"]),
        hovertemplate="City:%{x}<br>Cuisine:%{y}<br>Avg Rating:%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_lay(), title="Avg Rating Heatmap  —  Cuisine × City")
    return fig

def ch_price_elasticity(fdf):
    """Average rating binned by cost brackets."""
    bins  = [0,300,700,1200,2000,9999]
    lbls  = ["≤300","301-700","701-1200","1201-2000","2000+"]
    fdf2  = fdf.copy()
    fdf2["Cost Bin"] = pd.cut(fdf2["Cost"], bins=bins, labels=lbls)
    d = fdf2.groupby("Cost Bin", observed=True).agg(
        avg_rating=("Rating","mean"), count=("Name","count")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Cost Bin"], y=d["count"],
                         name="Count", marker_color=C_BLUE,
                         yaxis="y", opacity=0.6))
    fig.add_trace(go.Scatter(x=d["Cost Bin"], y=d["avg_rating"],
                              name="Avg Rating", mode="lines+markers",
                              marker=dict(color=C_GREEN, size=8),
                              line=dict(color=C_GREEN, width=2),
                              yaxis="y2"))
    # Apply base layout first, then override axes separately to avoid
    # duplicate keyword argument error (_lay() already contains 'yaxis')
    fig.update_layout(**_lay())
    fig.update_layout(
        title="Price Elasticity  —  Cost Bracket vs Rating",
        yaxis=dict(
            title="Restaurant Count",
            gridcolor=T()["border"],
            tickfont=dict(color=T()["muted"]),
        ),
        yaxis2=dict(
            title="Avg Rating",
            overlaying="y",
            side="right",
            gridcolor=T()["border"],
            tickfont=dict(color=T()["muted"]),
        ),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ── TAB 3: RANKINGS ───────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def ch_top10_rated(fdf):
    top = (fdf[fdf["Votes"]>=30]
           .sort_values(["Rating","Votes"], ascending=False).head(10))
    fig = px.bar(top, x="Rating", y="Name", orientation="h",
                 color="Rating",
                 color_continuous_scale=[C_RED,C_AMBER,C_GREEN],
                 hover_data=["City","Votes","Cost"],
                 title="Top 10 — Highest Rated", text="Rating")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                      marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return W(fig)

def ch_top10_votes(fdf):
    top = fdf.sort_values("Votes", ascending=False).head(10)
    fig = px.bar(top, x="Votes", y="Name", orientation="h",
                 color="Votes",
                 color_continuous_scale=[C_BLUE,C_PURPLE],
                 hover_data=["City","Rating","Cost"],
                 title="Top 10 — Most Reviewed", text="Votes")
    fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                      marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return W(fig)

def ch_underrated(fdf):
    """High rating, low votes — hidden gems."""
    med_votes = fdf["Votes"].median()
    gems = (fdf[(fdf["Rating"] >= 4.0) & (fdf["Votes"] <= med_votes)]
            .sort_values("Rating", ascending=False).head(10))
    fig = px.scatter(gems, x="Votes", y="Rating",
                     size="Cost", color="City",
                     hover_name="Name",
                     hover_data=["Primary Cuisine","Cost","Votes"],
                     title="Hidden Gems  —  High Rating, Low Votes",
                     color_discrete_sequence=PALETTE)
    fig.add_vline(x=med_votes, line_dash="dash",
                  line_color=T()["muted"],
                  annotation_text="Median votes",
                  annotation_font_color=T()["muted"])
    return W(fig)

def ch_cuisine_rating(fdf):
    cr = (fdf.groupby("Primary Cuisine")
          .agg(avg_rating=("Rating","mean"), count=("Name","count"))
          .query("count >= 10")
          .sort_values("avg_rating", ascending=False)
          .head(15).reset_index())
    fig = px.bar(cr, x="avg_rating", y="Primary Cuisine", orientation="h",
                 color="avg_rating",
                 color_continuous_scale=[C_RED,C_AMBER,C_GREEN],
                 title="Avg Rating by Cuisine  (min 10 restaurants)",
                 text="avg_rating")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                      marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return W(fig)

def percentile_table(fdf):
    """Top-20 restaurants with percentile rank."""
    cols = ["Name","City","Primary Cuisine","Rating","Votes","Cost","Rating Pct"]
    d = (fdf[cols].sort_values("Rating Pct", ascending=False)
         .head(20).reset_index(drop=True))
    d.index += 1
    d.columns = ["Name","City","Cuisine","Rating","Votes","Cost (₹)","Rating Percentile"]
    return d

def comparison_chart(fdf, names):
    sub = fdf[fdf["Name"].isin(names)].drop_duplicates("Name")
    if sub.empty:
        return None
    dims = ["Rating","Votes","Cost"]
    fig = go.Figure()
    colors = PALETTE
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [row[d] for d in dims]
        # Normalise to 0-1 for radar
        norms = []
        for d, v in zip(dims, vals):
            lo, hi = fdf[d].min(), fdf[d].max()
            norms.append((v - lo) / (hi - lo + 1e-9))
        norms.append(norms[0])
        fig.add_trace(go.Scatterpolar(
            r=norms + [norms[0]],
            theta=dims + [dims[0]],
            fill="toself", opacity=0.6,
            name=row["Name"][:20],
            line=dict(color=colors[i % len(colors)]),
        ))
    fig.update_layout(**_lay(),
        title="Restaurant Comparison  (normalised radar)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color=T()["muted"],
                            gridcolor=T()["border"]),
            angularaxis=dict(color=T()["muted"],
                             gridcolor=T()["border"])))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ── TAB 4: INSIGHTS ───────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def ch_sentiment(fdf):
    pos = (fdf["Rating"] >= 4.0).sum()
    neu = ((fdf["Rating"] >= 3.0) & (fdf["Rating"] < 4.0)).sum()
    neg = (fdf["Rating"] < 3.0).sum()
    total = pos+neu+neg
    fig = go.Figure(go.Bar(
        x=["Positive","Neutral","Negative"], y=[pos,neu,neg],
        marker_color=[C_GREEN,C_AMBER,C_RED],
        text=[f"{v:,}  ({v/total*100:.1f}%)" for v in [pos,neu,neg]],
        textposition="outside",
        textfont=dict(color=T()["text"], size=10),
    ))
    fig.update_layout(**_lay(), title="Review Sentiment  (rating-based proxy)",
                      showlegend=False)
    return fig

def ch_delivery_booking(fdf):
    fig = make_subplots(1, 2, specs=[[{"type":"pie"},{"type":"pie"}]],
                        subplot_titles=["Online Delivery","Table Booking"])
    od = fdf["Online Delivery"].value_counts()
    tb = fdf["Table Booking"].value_counts()
    fig.add_trace(go.Pie(labels=od.index, values=od.values, hole=0.52,
                         marker_colors=[C_GREEN,T()["border"]],
                         textfont=dict(size=10)), 1, 1)
    fig.add_trace(go.Pie(labels=tb.index, values=tb.values, hole=0.52,
                         marker_colors=[C_BLUE,T()["border"]],
                         textfont=dict(size=10)), 1, 2)
    fig.update_layout(**_lay(), title_text="Service Availability")
    for ann in fig.layout.annotations:
        ann.font.color = T()["muted"]
        ann.font.size  = 11
    return fig

def ch_votes_hist(fdf):
    fig = px.histogram(fdf, x="Votes", nbins=40,
                       color_discrete_sequence=[C_PURPLE],
                       marginal="violin",
                       title="Votes Distribution")
    fig.update_traces(marker_line_width=0, opacity=0.85)
    return W(fig)

def ch_rating_stack(fdf):
    ORDER = ["Excellent  4.5+","Very Good  4.0–4.4",
             "Good  3.5–3.9","Average  3.0–3.4","Below Avg  <3.0"]
    top10 = fdf["City"].value_counts().nlargest(10).index
    g = (fdf[fdf["City"].isin(top10)]
         .groupby(["City","Rating Band"]).size().reset_index(name="Count"))
    fig = px.bar(g, x="City", y="Count", color="Rating Band",
                 barmode="stack",
                 category_orders={"Rating Band": ORDER},
                 color_discrete_sequence=[C_GREEN,C_BLUE,C_AMBER,C_PURPLE,C_RED],
                 title="Rating Bands by City  (Top 10)")
    return W(fig)

def ch_type_donut(fdf):
    d = fdf["Restaurant Type"].value_counts().reset_index()
    d.columns = ["Type","Count"]
    fig = px.pie(d, names="Type", values="Count",
                 hole=0.50, title="Restaurant Type Breakdown",
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textfont=dict(size=10), textinfo="label+percent")
    return W(fig)

def ch_cost_violin(fdf):
    top_types = fdf["Restaurant Type"].value_counts().nlargest(5).index
    sub = fdf[fdf["Restaurant Type"].isin(top_types)]
    fig = px.violin(sub, x="Restaurant Type", y="Cost",
                    color="Restaurant Type", box=True,
                    color_discrete_sequence=PALETTE,
                    title="Cost Distribution by Restaurant Type")
    fig.update_layout(showlegend=False)
    return W(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ── TAB 5: MAP & DATA ─────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def render_map(fdf):
    s = fdf.sample(min(800, len(fdf)), random_state=42).dropna(
        subset=["Latitude","Longitude"])
    try:
        import folium
        from streamlit_folium import st_folium
        tile = "CartoDB dark_matter" if st.session_state.dark_mode else "CartoDB positron"
        m = folium.Map(location=[s["Latitude"].mean(), s["Longitude"].mean()],
                       zoom_start=5, tiles=tile)
        for _, r in s.iterrows():
            color = (C_GREEN if r["Rating"]>=4.0
                     else C_AMBER if r["Rating"]>=3.0 else C_RED)
            folium.CircleMarker(
                location=[r["Latitude"], r["Longitude"]],
                radius=max(4, min(10, r["Votes"]/400)),
                color=color, fill=True, fill_opacity=0.75,
                tooltip=(f"<b>{r['Name']}</b><br>"
                         f"{r['City']} · {r.get('Primary Cuisine','')}<br>"
                         f"Rating {r['Rating']} · ₹{int(r['Cost'])} · {r['Votes']} votes"),
            ).add_to(m)
        st_folium(m, height=480, use_container_width=True)
    except ModuleNotFoundError:
        fig = px.scatter_mapbox(
            s, lat="Latitude", lon="Longitude",
            color="Rating", size="Votes", size_max=18,
            hover_name="Name",
            hover_data={"City":True,"Primary Cuisine":True,"Cost":True,"Votes":True},
            color_continuous_scale=[C_RED,C_AMBER,C_GREEN],
            mapbox_style="carto-darkmatter" if st.session_state.dark_mode else "carto-positron",
            zoom=4, height=480, title="Restaurant Locations",
        )
        fig.update_layout(**_lay())
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Green ≥ 4.0  ·  Amber 3.0–3.9  ·  Red < 3.0  ·  "
               f"Showing {len(s):,} of {len(fdf):,} restaurants")

def render_table(fdf):
    with st.expander("View & download filtered dataset", expanded=False):
        COLS = ["Name","City","Locality","Primary Cuisine","Restaurant Type",
                "Rating","Rating Band","Rating Pct","Votes","Cost","Price Tier",
                "Online Delivery","Table Booking"]
        COLS = [c for c in COLS if c in fdf.columns]

        def _cr(val):
            try:
                v = float(val)
                if v>=4.5: return f"color:{C_GREEN};font-weight:600"
                if v>=4.0: return f"color:{C_BLUE};font-weight:600"
                if v>=3.5: return f"color:{C_AMBER}"
                return f"color:{C_RED}"
            except: return ""

        styled = fdf[COLS].style.applymap(_cr, subset=["Rating"])
        st.dataframe(styled, use_container_width=True, height=400)
        csv = fdf[COLS].to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered data  (CSV)", csv,
                           file_name="restaurants_filtered.csv",
                           mime="text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    DEFAULT = "restaurants.csv"

    # ── Route: if no dataset chosen yet, show landing page ─────────────────
    # "upload" mode stays on landing until the user clicks "Open Dashboard →"
    # which sets uploaded_df in session_state and reruns into dashboard mode.
    choice = st.session_state.dataset_choice
    if choice is None or (choice == "upload" and st.session_state.uploaded_df is None):
        landing_page()
        st.stop()

    inject_css()
    t = T()

    # ── Load default ───────────────────────────────────────────────────────
    try:
        df_default = load_default(DEFAULT)
    except FileNotFoundError:
        st.error("restaurants.csv not found in the app folder.")
        st.stop()

    # ── Resolve active dataset BEFORE building sidebar ─────────────────────
    # BUG FIX: previously sidebar() always received df_default, so city/cuisine
    # dropdowns and cost/rating sliders were always wrong for uploaded datasets.
    # Also, the old else-branch wiped uploaded_df whenever the sidebar uploader
    # was empty — destroying data loaded from the landing page uploader.
    if (st.session_state.dataset_choice == "upload"
            and st.session_state.uploaded_df is not None
            and not st.session_state.upload_error):
        base_df = st.session_state.uploaded_df
    else:
        base_df = df_default

    # ── Sidebar (filters + dropdowns built from the active dataset) ────────
    (sidebar_upload, search, city, cuisine, r_type,
     rating_rng, cost_rng, min_votes,
     online_opt, table_opt) = sidebar(base_df)

    # ── Handle a NEW file dropped into the sidebar uploader ────────────────
    # Only call .read() once per unique file_id — never on every rerun.
    if sidebar_upload is not None:
        file_id = (getattr(sidebar_upload, "file_id", None)
                   or f"{sidebar_upload.name}_{sidebar_upload.size}")
        if file_id != st.session_state.file_id:
            st.session_state.upload_error = None
            try:
                with st.spinner(f"Processing {sidebar_upload.name} …"):
                    raw = sidebar_upload.read()
                    raw_df_for_info = pd.read_csv(io.BytesIO(raw), low_memory=False)
                    st.session_state.col_info       = detect_columns(raw_df_for_info)
                    del raw_df_for_info
                    st.session_state.uploaded_df    = load_uploaded(raw)
                    st.session_state.file_id        = file_id
                    st.session_state.dataset_choice = "upload"
                    del raw
                st.toast(
                    f"Loaded {sidebar_upload.name} — "
                    f"{len(st.session_state.uploaded_df):,} rows", icon="✅")
                st.rerun()   # rebuild sidebar with correct filter ranges
            except Exception as e:
                st.session_state.upload_error = str(e)
                st.session_state.uploaded_df  = None
                st.session_state.file_id      = None

    # ── Final df: uploaded if available, otherwise default ─────────────────
    if st.session_state.upload_error:
        st.error(f"Could not load file: {st.session_state.upload_error}\n\n"
                 "CSV must have: Name, City, Cuisine, Rating, Votes, Cost.",
                 icon="🚫")
        df = df_default
    elif (st.session_state.dataset_choice == "upload"
          and st.session_state.uploaded_df is not None):
        df = st.session_state.uploaded_df
    else:
        df = df_default

    # ── Filter ─────────────────────────────────────────────────────────────
    fdf = apply_filters(df, search, city, cuisine, r_type,
                        rating_rng, cost_rng, min_votes, online_opt, table_opt)

    # ── Header ─────────────────────────────────────────────────────────────
    theme_badge = ("🌙 Dark" if st.session_state.dark_mode else "☀️ Light")
    st.markdown(f"""
    <div style="padding:20px 0 6px;display:flex;justify-content:space-between;
                align-items:flex-end">
        <div>
            <div class="page-title">Restaurant Analytics Dashboard</div>
            <div class="page-sub">
                {len(fdf):,} restaurants · {df['City'].nunique()} cities ·
                {theme_badge} mode
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if fdf.empty:
        st.warning("No restaurants match the selected filters.")
        return

    # ── KPIs ───────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>Key Metrics</div>", unsafe_allow_html=True)
    kpi_cards(fdf)

    # ══════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Overview  ",
        "  Advanced Analytics  ",
        "  Rankings  ",
        "  Insights  ",
        "  Map & Data  ",
    ])

    # ── Tab 1: Overview ────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='sec-title'>City & Cuisine</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_city_bar(fdf),     use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_cuisine_donut(fdf),use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Ratings & Pricing</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_rating_hist(fdf),  use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_price_bar(fdf),    use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_scatter(fdf),      use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c6:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_city_cost(fdf),    use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Advanced Analytics ──────────────────────────────────────────
    with tab2:
        st.markdown("<div class='sec-title'>Hierarchy & Composition</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.plotly_chart(ch_treemap(fdf), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        c7, c8 = st.columns(2)
        with c7:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_sunburst(fdf),  use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c8:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_heatmap(fdf),   use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Distribution & Correlation</div>",
                    unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        with c9:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_boxplot(fdf),   use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c10:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_violin(fdf),    use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.plotly_chart(ch_parallel(fdf), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.plotly_chart(ch_price_elasticity(fdf), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: Rankings ────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='sec-title'>Top Performers</div>",
                    unsafe_allow_html=True)
        c11, c12 = st.columns(2)
        with c11:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_top10_rated(fdf), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c12:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_top10_votes(fdf), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Hidden Gems & Cuisine Rankings</div>",
                    unsafe_allow_html=True)
        c13, c14 = st.columns(2)
        with c13:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_underrated(fdf),  use_container_width=True)
            st.markdown("""<div class='insight-box'>
                Restaurants with rating ≥ 4.0 but votes below the dataset median.
                These are highly rated but less discovered — ideal for exploration.
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c14:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_cuisine_rating(fdf), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Percentile Rankings  (Top 20)</div>",
                    unsafe_allow_html=True)
        pct_df = percentile_table(fdf)

        def _cr2(val):
            try:
                v = float(val)
                if v>=4.5: return f"color:{C_GREEN};font-weight:600"
                if v>=4.0: return f"color:{C_BLUE};font-weight:600"
                if v>=3.5: return f"color:{C_AMBER}"
                return f"color:{C_RED}"
            except: return ""

        def _crp(val):
            try:
                v = float(val)
                if v>=90: return f"color:{C_GREEN};font-weight:600"
                if v>=75: return f"color:{C_BLUE}"
                return ""
            except: return ""

        styled_pct = (pct_df.style
                      .applymap(_cr2,  subset=["Rating"])
                      .applymap(_crp,  subset=["Rating Percentile"]))
        st.dataframe(styled_pct, use_container_width=True, height=440)

        st.markdown("<div class='sec-title'>Restaurant Comparison Tool</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='insight-box'>Select up to 5 restaurants to compare "
                    "their Rating, Votes and Cost on a radar chart.</div>",
                    unsafe_allow_html=True)
        all_names = sorted(fdf["Name"].dropna().unique().tolist())
        chosen = st.multiselect("Select restaurants to compare",
                                all_names, max_selections=5,
                                placeholder="Type to search…")
        if chosen:
            fig_comp = comparison_chart(fdf, chosen)
            if fig_comp:
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.plotly_chart(fig_comp, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Select at least one restaurant above to see the radar comparison.")

    # ── Tab 4: Insights ────────────────────────────────────────────────────
    with tab4:
        st.markdown("<div class='sec-title'>Sentiment & Engagement</div>",
                    unsafe_allow_html=True)
        c15, c16 = st.columns(2)
        with c15:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_sentiment(fdf),       use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c16:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_delivery_booking(fdf),use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Votes & Type Analysis</div>",
                    unsafe_allow_html=True)
        c17, c18 = st.columns(2)
        with c17:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_votes_hist(fdf),  use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c18:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_type_donut(fdf),  use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c19, c20 = st.columns(2)
        with c19:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_rating_stack(fdf),use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c20:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(ch_cost_violin(fdf), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 5: Map & Data ──────────────────────────────────────────────────
    with tab5:
        st.markdown("<div class='sec-title'>Location Map</div>",
                    unsafe_allow_html=True)
        render_map(fdf)
        st.markdown("<div class='sec-title'>Data Explorer</div>",
                    unsafe_allow_html=True)
        render_table(fdf)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-bar">
        Restaurant Analytics Dashboard &nbsp;·&nbsp;
        Streamlit &nbsp;·&nbsp; Plotly
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
