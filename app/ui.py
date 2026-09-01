"""BhoomiSetu UI design system (pure Streamlit/Python — no Node).

Provides the shared brand palette, the "BS" monogram (inline SVG), global CSS, a
branded Plotly template, and reusable styled components (hero, KPI cards, risk
badges, section/footer). Import and use from app/streamlit_app.py.
"""

from __future__ import annotations

import plotly.io as pio
import plotly.graph_objects as go  # noqa: F401
import streamlit as st

# --- palette (locked) ------------------------------------------------------ #
NAVY = "#1F4E79"
STEEL = "#4A90A4"
GRN = "#2E7D32"
YEL = "#E8A33D"
RED = "#C62828"
BG = "#F5F7FA"
TEXT = "#1A1A1A"

COLORS = {"RED": RED, "YELLOW": YEL, "GREEN": GRN}
EMOJI = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}

# --- dark palette ----------------------------------------------------------- #
DARK_BG = "#0F1B2D"
DARK_CARD = "#16283E"
DARK_SIDEBAR = "#0B1626"
DARK_TEXT = "#E5EAF0"
DARK_MUTED = "#93A3B5"
DARK_GRID = "#26364F"
DARK_HEADING = "#A9C4E8"

NAME = "BhoomiSetu"
TAGLINE = "Predicting delays before they cost the project."
SUB = "Bhoomi = land · Setu = bridge"
STATES = ["HP", "PB", "UK"]

MONOGRAM = f"""
<svg width="42" height="42" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="21" cy="21" r="20" fill="{NAVY}"/>
  <circle cx="21" cy="21" r="20" fill="none" stroke="{STEEL}" stroke-width="1.5"/>
  <text x="21" y="27" text-anchor="middle" font-family="Arial, Helvetica, sans-serif"
        font-size="16" font-weight="700" fill="#ffffff">BS</text>
</svg>
"""

_CSS = f"""
<style>
:root {{ --navy:{NAVY}; --steel:{STEEL}; --grn:{GRN}; --yel:{YEL}; --red:{RED}; }}
.stApp {{ background-color:{BG}; }}
h3 {{ color:{NAVY}; }}

/* sidebar brand band */
.bk-brand {{ background: linear-gradient(135deg, {NAVY} 0%, #2C6B9B 60%, {STEEL} 100%);
            padding:12px 14px; border-radius:10px; margin-bottom:10px; }}
.bk-brand .name {{ color:#fff; font-size:1.06rem; font-weight:800; letter-spacing:.02em; line-height:1.15; }}
.bk-brand .tag {{ color:rgba(255,255,255,.85); font-size:.7rem; }}

/* hero */
.bk-hero {{ background: linear-gradient(135deg, {NAVY} 0%, #2C6B9B 55%, {STEEL} 100%);
            color:#fff; padding:20px 26px; border-radius:12px; margin-bottom:14px; }}
.bk-hero h1 {{ margin:0 0 2px 0; font-size:1.55rem; color:#fff; }}
.bk-hero .tag {{ font-size:.95rem; opacity:.92; }}
.bk-hero .sub {{ font-size:.78rem; opacity:.75; margin-top:4px; }}

/* KPI cards */
.bk-kpi {{ background:#fff; border-radius:10px; padding:12px 16px;
          border-left:4px solid {NAVY}; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.bk-kpi .k-label {{ font-size:.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }}
.bk-kpi .k-value {{ font-size:1.45rem; font-weight:800; color:{NAVY}; line-height:1.25; }}
.bk-kpi .k-delta {{ font-size:.82rem; color:#555; }}

/* risk badge */
.bk-badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
            font-size:.72rem; font-weight:700; color:#fff; }}

/* section + footer */
.bk-section {{ color:{NAVY}; font-weight:800; margin:.5rem 0 .2rem; font-size:1.04rem; }}
.bk-footer {{ margin-top:26px; padding-top:10px; border-top:1px solid #e2e8f0;
             color:#6b7280; font-size:.78rem; }}
</style>
"""

_DARK_CSS = f"""
<style>
:root {{ --bg:{DARK_BG}; --card:{DARK_CARD}; --txt:{DARK_TEXT}; }}
.stApp {{ background-color:{DARK_BG} !important; }}
[data-testid="stAppViewContainer"] {{ background-color:{DARK_BG}; }}
[data-testid="stSidebar"] {{ background-color:{DARK_SIDEBAR}; }}
[data-testid="stSidebar"] * {{ color:{DARK_TEXT} !important; }}
h3 {{ color:{DARK_HEADING}; }}
.bk-kpi {{ background:{DARK_CARD}; box-shadow:0 1px 3px rgba(0,0,0,.35); }}
.bk-kpi .k-label {{ color:{DARK_MUTED}; }}
.bk-kpi .k-value {{ color:{DARK_TEXT}; }}
.bk-kpi .k-delta {{ color:{DARK_MUTED}; }}
.bk-section {{ color:{DARK_HEADING}; }}
.bk-footer {{ color:{DARK_MUTED}; border-top:1px solid {DARK_GRID}; }}
[data-testid="stDataFrame"] {{ background:{DARK_CARD}; }}
[data-testid="stDataFrame"] table, [data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td {{ color:{DARK_TEXT}; }}
[data-testid="stDataFrame"] thead tr th {{ background:{DARK_SIDEBAR}; }}
[data-testid="stExpander"] {{ background:{DARK_CARD}; }}
</style>
"""


def inject_css(theme: str = "Light") -> None:
    st.markdown(_CSS + (_DARK_CSS if theme == "Dark" else ""), unsafe_allow_html=True)


def setup_plotly(theme: str = "Light") -> None:
    """Register the branded 'sih' template (theme-aware) and make it the default."""
    dark = theme == "Dark"
    grid = DARK_GRID if dark else "#E8EDF2"
    zero = "#3A4C66" if dark else "#D5DCE4"
    font_color = DARK_TEXT if dark else TEXT
    t = go.layout.Template()
    t.layout = dict(
        font=dict(family="'Segoe UI','Helvetica Neue',Arial,sans-serif", size=13, color=font_color),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[NAVY, STEEL, YEL, GRN, RED, "#8E44AD"],
        xaxis=dict(gridcolor=grid, zerolinecolor=zero),
        yaxis=dict(gridcolor=grid, zerolinecolor=zero),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    pio.templates["sih"] = t
    pio.templates.default = "sih"


def brand_sidebar() -> None:
    st.markdown(
        f'<div class="bk-brand">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'{MONOGRAM}'
        f'<div><div class="name">{NAME}</div><div class="tag">{TAGLINE}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True)


def hero(title: str, tagline: str = TAGLINE, sub: str | None = None) -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="bk-hero"><h1>{title}</h1><div class="tag">{tagline}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str | None = None, color: str = NAVY) -> None:
    delta_html = f'<div class="k-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="bk-kpi" style="border-left-color:{color}">'
        f'<div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True)


def kpi_row(cards: list[tuple[str, str, str | None, str]]) -> None:
    """cards = [(label, value, delta, color), ...] rendered in one row."""
    cols = st.columns(len(cards))
    for col, (label, value, delta, color) in zip(cols, cards):
        with col:
            kpi_card(label, value, delta, color)


def risk_badge(level: str) -> str:
    color = COLORS.get(level, "#888888")
    return (f'<span class="bk-badge" style="background:{color}">'
            f'{EMOJI.get(level, "")} {level}</span>')


def section(text: str) -> None:
    st.markdown(f'<div class="bk-section">{text}</div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="bk-footer">{text}</div>', unsafe_allow_html=True)
