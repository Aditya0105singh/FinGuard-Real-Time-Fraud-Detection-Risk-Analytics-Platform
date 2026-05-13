"""FinGuard design system.

All HTML components use inline styles so they render correctly even if
class-based CSS is stripped by Streamlit. CSS overrides handle sidebar,
metric cards, tabs, and other native widgets.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Palette ────────────────────────────────────────────────────────────
BLUE          = "#2563EB"
BLUE_DARK     = "#1D4ED8"
BLUE_LIGHT    = "#EFF6FF"
BLUE_BORDER   = "#BFDBFE"

NAVY          = "#0F172A"
NAVY_2        = "#1E293B"

SLATE_900     = "#0F172A"
SLATE_700     = "#334155"
SLATE_600     = "#475569"
SLATE_500     = "#64748B"
SLATE_400     = "#94A3B8"
SLATE_200     = "#E2E8F0"
SLATE_100     = "#F1F5F9"
SLATE_50      = "#F8FAFC"

GREEN         = "#059669"
GREEN_BG      = "#ECFDF5"
GREEN_BORDER  = "#A7F3D0"
AMBER         = "#D97706"
ORANGE        = "#EA580C"
RED           = "#DC2626"
RED_BG        = "#FEF2F2"

# aliases kept for app.py imports
INDIGO        = BLUE
INDIGO_DARK   = BLUE_DARK

RISK_COLORS   = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": ORANGE, "CRITICAL": RED}
RISK_BG       = {"LOW": GREEN_BG, "MEDIUM": "#FFFBEB", "HIGH": "#FFF7ED", "CRITICAL": RED_BG}

CHART_SEQ     = [BLUE, "#0EA5E9", "#7C3AED", GREEN, AMBER, RED]


# ── CSS (native widget overrides only) ────────────────────────────────
_CSS = f"""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 sans-serif !important;
}}

/* ── Global ──────────────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}

[data-testid="stAppViewContainer"],
.stApp {{ background-color: {SLATE_50} !important; }}

[data-testid="block-container"] {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {NAVY} !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}}
[data-testid="stSidebar"] > div {{
    background-color: {NAVY} !important;
}}
[data-testid="stSidebarContent"] {{
    background-color: {NAVY} !important;
}}

/* sidebar text colours */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] code {{
    color: {SLATE_400} !important;
}}
[data-testid="stSidebar"] h1 {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] hr {{
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
}}

/* sidebar radio → pill navigation */
[data-testid="stSidebar"] [role="radiogroup"] > label {{
    display: block !important;
    padding: 0.48rem 0.8rem !important;
    margin: 0.1rem 0 !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
    background: rgba(37,99,235,0.18) !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background: {BLUE} !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4) !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) span,
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) div {{
    color: #FFFFFF !important;
    font-weight: 600 !important;
}}
/* hide the radio dot */
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
    display: none !important;
}}

/* ── Metric cards ────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: #FFFFFF !important;
    border: 1px solid {SLATE_200} !important;
    border-top: 3px solid {BLUE} !important;
    border-radius: 12px !important;
    padding: 1rem 1.1rem 0.85rem !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.06) !important;
}}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] label {{
    color: {SLATE_500} !important;
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}
[data-testid="stMetricValue"] > div {{
    color: {SLATE_900} !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.8rem !important;
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stFormSubmitButton > button {{
    background: {BLUE} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.3) !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {{
    background: {BLUE_DARK} !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
}}

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {SLATE_200} !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    color: {SLATE_600} !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    padding: 0.38rem 0.9rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: #FFFFFF !important;
    color: {BLUE} !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.1) !important;
}}

/* ── Expanders ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: #FFFFFF !important;
    border: 1px solid {SLATE_200} !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04) !important;
}}
[data-testid="stExpander"] summary {{
    color: {SLATE_700} !important;
    font-weight: 600 !important;
}}

/* ── DataFrames ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {SLATE_200} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}

/* ── Form / number input ─────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    border-radius: 8px !important;
    border: 1px solid {SLATE_200} !important;
    background: #FFFFFF !important;
    color: {SLATE_900} !important;
}}

/* ── Info / warning / error boxes ───────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 10px !important;
}}

/* ── Captions ────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {{
    color: {SLATE_500} !important;
    font-size: 0.82rem !important;
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Plotly template ────────────────────────────────────────────────────
def register_plotly_template() -> None:
    pio.templates["finguard"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, sans-serif", color=SLATE_700, size=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            colorway=CHART_SEQ,
            margin=dict(t=36, r=20, b=48, l=60),
            xaxis=dict(
                gridcolor=SLATE_200, gridwidth=1,
                linecolor=SLATE_200, zerolinecolor=SLATE_200,
                tickfont=dict(size=11, color=SLATE_500),
            ),
            yaxis=dict(
                gridcolor=SLATE_200, gridwidth=1,
                linecolor=SLATE_200, zerolinecolor=SLATE_200,
                tickfont=dict(size=11, color=SLATE_500),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=SLATE_700),
            ),
            hoverlabel=dict(
                bgcolor="#FFFFFF", bordercolor=SLATE_200,
                font=dict(family="Inter, sans-serif", size=12, color=SLATE_900),
            ),
        )
    )
    pio.templates.default = "finguard"


# ── HTML components (all inline-styled) ───────────────────────────────

def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chips_html = "".join(
        f'<span style="display:inline-block;background:rgba(255,255,255,0.12);'
        f'border:1px solid rgba(255,255,255,0.22);border-radius:999px;'
        f'padding:0.2rem 0.7rem;margin:0 0.28rem 0.28rem 0;'
        f'font-size:0.74rem;font-weight:600;color:#DBEAFE;">{c}</span>'
        for c in chips
    )
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{NAVY} 0%,#0F3460 55%,{BLUE} 100%);
                    border-radius:16px;padding:2.2rem 2.5rem;margin-bottom:1.4rem;
                    position:relative;overflow:hidden;">
          <div style="position:absolute;top:-60px;right:-60px;width:300px;height:300px;
                      background:radial-gradient(circle,rgba(37,99,235,0.3) 0%,transparent 70%);
                      pointer-events:none;"></div>
          <h1 style="color:#FFFFFF;font-size:2.1rem;font-weight:800;
                     margin:0 0 0.45rem 0;letter-spacing:-0.03em;">{title}</h1>
          <p style="color:#BFDBFE;font-size:1rem;max-width:48rem;
                    margin:0;line-height:1.65;">{subtitle}</p>
          <div style="margin-top:1.1rem;">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    cap_html = (
        f'<p style="color:{SLATE_500};font-size:0.88rem;'
        f'margin:0.3rem 0 0 0.5rem;line-height:1.5;">{caption}</p>'
        if caption else ""
    )
    st.markdown(
        f"""
        <div style="margin:1.6rem 0 0.65rem 0;padding-bottom:0.55rem;
                    border-bottom:2px solid {SLATE_200};">
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <div style="width:4px;height:1.15em;background:{BLUE};
                        border-radius:4px;flex-shrink:0;"></div>
            <h2 style="font-size:1.15rem;font-weight:700;margin:0;
                       color:{SLATE_900};letter-spacing:-0.02em;">{title}</h2>
          </div>
          {cap_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid {SLATE_200};
                    border-radius:12px;padding:1.25rem 1.3rem;height:100%;
                    box-shadow:0 1px 4px rgba(15,23,42,0.05);">
          <div style="font-size:1.55rem;margin-bottom:0.55rem;">{icon}</div>
          <h4 style="margin:0 0 0.4rem 0;font-size:0.97rem;font-weight:700;
                     color:{SLATE_900};">{title}</h4>
          <p style="margin:0;color:{SLATE_600};font-size:0.86rem;
                    line-height:1.55;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(index: int, title: str, finding: str, recommendation: str) -> None:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid {SLATE_200};
                    border-left:4px solid {BLUE};border-radius:10px;
                    padding:1.15rem 1.4rem;margin-bottom:0.75rem;
                    box-shadow:0 1px 4px rgba(15,23,42,0.04);">
          <h4 style="margin:0 0 0.5rem 0;color:{SLATE_900};
                     font-size:1rem;font-weight:700;">{index}. {title}</h4>
          <p style="margin:0 0 0.65rem 0;color:{SLATE_700};
                    line-height:1.6;font-size:0.89rem;">{finding}</p>
          <div style="background:{GREEN_BG};border:1px solid {GREEN_BORDER};
                      color:#065F46;border-radius:8px;padding:0.5rem 0.85rem;
                      font-size:0.86rem;display:flex;gap:0.4rem;
                      align-items:flex-start;">
            <span>✅</span>
            <span><strong>Recommendation:</strong> {recommendation}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_banner(risk_level: str, recommendation: str) -> None:
    bg   = RISK_COLORS.get(risk_level, SLATE_500)
    icon = "🚨" if risk_level in ("HIGH", "CRITICAL") else "✅"
    st.markdown(
        f"""
        <div style="background:{bg};border-radius:12px;
                    padding:1.2rem 1.5rem;margin:0.5rem 0 1.1rem 0;
                    position:relative;overflow:hidden;">
          <div style="position:absolute;inset:0;
                      background:linear-gradient(135deg,rgba(255,255,255,0.15) 0%,
                      transparent 60%);pointer-events:none;"></div>
          <h2 style="color:#FFFFFF;margin:0;font-size:1.5rem;
                     font-weight:800;position:relative;">
            {icon} {risk_level} RISK</h2>
          <p style="color:rgba(255,255,255,0.92);margin:0.3rem 0 0 0;
                    font-size:0.97rem;position:relative;">{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step(num: int, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:flex-start;gap:0.8rem;
                    padding:0.65rem 0.9rem;margin-bottom:0.5rem;
                    background:#FFFFFF;border:1px solid {SLATE_200};
                    border-radius:10px;box-shadow:0 1px 3px rgba(15,23,42,0.04);">
          <div style="background:{BLUE};color:#FFFFFF;min-width:1.7rem;
                      height:1.7rem;border-radius:999px;display:flex;
                      align-items:center;justify-content:center;
                      font-weight:700;font-size:0.82rem;flex-shrink:0;
                      box-shadow:0 2px 6px rgba(37,99,235,0.35);">{num}</div>
          <p style="margin:0.15rem 0 0 0;color:{SLATE_700};
                    font-size:0.9rem;line-height:1.45;">
            <strong style="color:{SLATE_900};">{title}</strong> — {body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
