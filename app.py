"""Kangemi Edu – Quantitative Boutique | Streamlit access gateway.

Rename the existing app.py to app_core.py and put this file alongside it.
"""

from __future__ import annotations

import hmac
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CORE_APP = BASE_DIR / "app_core.py"

st.set_page_config(page_title="Kangemi Edu | Quantitative Boutique", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")


def _authenticated() -> bool:
    return bool(st.session_state.get("kangemi_authenticated", False))


def _check_password(candidate: str) -> bool:
    try:
        expected = st.secrets["APP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        st.error("Password non configurata. Imposta APP_PASSWORD nei Secrets di Streamlit Cloud.")
        return False
    return bool(candidate) and hmac.compare_digest(candidate.encode("utf-8"), str(expected).encode("utf-8"))


def _login_screen() -> None:
    st.markdown("""
    <style>
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none!important}
      [data-testid="stHeader"] {background:transparent!important}
      [data-testid="stAppViewContainer"] {
        background:radial-gradient(ellipse at 15% 18%,rgba(71,58,139,.22),transparent 39%),
                   radial-gradient(ellipse at 83% 72%,rgba(18,128,145,.15),transparent 37%),#080d19;
      }
      .block-container {max-width:1220px;padding-top:2.2rem;padding-bottom:1.5rem}
      .stForm {background:rgba(13,21,38,.83);border:1px solid rgba(133,157,203,.23);
               border-radius:22px;padding:1.5rem 1.6rem;box-shadow:0 28px 80px rgba(0,0,0,.35)}
      .stForm label {color:#bdcce5!important}
      .stForm input {background:#0a1326!important;color:#fff!important;border:1px solid #344461!important}
      .stForm button {width:100%;background:linear-gradient(95deg,#6659d8,#318bba)!important;
                      color:white!important;border:0!important;border-radius:10px!important;font-weight:650!important}
      .stForm button:hover {filter:brightness(1.15)}
      .k-brand {font-size:.79rem;letter-spacing:.23em;font-weight:800;color:#c6cbff;text-transform:uppercase}
      .k-eyebrow {font-size:.75rem;letter-spacing:.18em;color:#71d8d6;text-transform:uppercase;margin-top:3rem}
      .k-title {font-size:clamp(2.8rem,5.2vw,5.1rem);line-height:1.07;font-weight:760;letter-spacing:-.045em;
                color:#f3f6ff;margin:.7rem 0 1.1rem}
      .k-title em {font-style:normal;background:linear-gradient(90deg,#a59aff,#70dfdf);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
      .k-sub {color:#a8b6cf;font-size:1.04rem;line-height:1.75;max-width:580px}
      .k-chips {display:flex;gap:10px;flex-wrap:wrap;margin:1.5rem 0}
      .k-chip {border:1px solid #34445e;background:#111a2d;color:#c3d3e7;padding:7px 12px;
               border-radius:40px;font-size:.72rem;letter-spacing:.04em}
      .k-panelhead {color:#fff;font-size:1.6rem;font-weight:720;margin:.3rem 0}
      .k-muted {color:#9faec8;font-size:.88rem;line-height:1.6}
      .k-mini {font-size:.7rem;letter-spacing:.12em;color:#72dbca}
      .k-footer {color:#657692;font-size:.74rem;text-align:center;margin-top:3rem}
      .k-chart {width:100%;margin-top:1.4rem;border:1px solid #24334b;border-radius:17px;
                background:linear-gradient(160deg,rgba(15,27,49,.9),rgba(11,18,34,.75));overflow:hidden}
      .k-line {stroke-dasharray:1200;stroke-dashoffset:1200;animation:k-draw 5.8s ease forwards,k-pulse 5s ease-in-out infinite 5.8s}
      .k-line2 {stroke-dasharray:1200;stroke-dashoffset:1200;animation:k-draw 7s ease forwards}
      .k-dot {animation:k-blink 2s ease-in-out infinite}
      .k-formula {animation:k-float 7s ease-in-out infinite}
      @keyframes k-draw {to {stroke-dashoffset:0}}
      @keyframes k-pulse {50% {opacity:.57}}
      @keyframes k-blink {50% {opacity:.25}}
      @keyframes k-float {50% {transform:translateY(-5px)}}
      @media(max-width:700px){.k-eyebrow{margin-top:1rem}.block-container{padding-top:1rem}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="k-brand">◈ &nbsp; KANGEMI EDU <span style="color:#8492b0">/ QUANTITATIVE BOUTIQUE</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1.22, .85], gap="large", vertical_alignment="center")

    with left:
        st.markdown("""
        <div class="k-eyebrow">Research · Portfolio Engineering · Risk Intelligence</div>
        <div class="k-title">Invest with<br><em>quantitative clarity.</em></div>
        <div class="k-sub">Un ambiente riservato per esplorare la teoria di portafoglio,
        confrontare strategie quantitative e studiare il rischio attraverso modelli,
        dati e simulazioni.</div>
        <div class="k-chips">
          <span class="k-chip">MARKOWITZ</span><span class="k-chip">FAMA–FRENCH</span>
          <span class="k-chip">GARCH</span><span class="k-chip">MONTE CARLO</span>
        </div>
        <div class="k-chart">
          <svg viewBox="0 0 700 295" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%" role="img" aria-label="Grafico finanziario stilizzato animato">
            <defs><linearGradient id="kfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#58d5c5" stop-opacity=".26"/><stop offset="1" stop-color="#58d5c5" stop-opacity="0"/></linearGradient></defs>
            <g stroke="#24334b" stroke-width="1" opacity=".75">
              <path d="M35 45H665 M35 95H665 M35 145H665 M35 195H665 M35 245H665"/>
              <path d="M85 25V260 M185 25V260 M285 25V260 M385 25V260 M485 25V260 M585 25V260"/>
            </g>
            <text x="30" y="23" fill="#a4b6d3" font-size="12" letter-spacing="2">QUANTITATIVE SIGNAL / ILLUSTRATIVE</text>
            <path d="M35 230 L72 213 L105 224 L142 181 L175 198 L211 157 L247 170 L285 138 L322 155 L360 112 L396 126 L435 91 L474 113 L512 74 L550 88 L589 50 L625 67 L665 37 L665 260 L35 260Z" fill="url(#kfill)"/>
            <path class="k-line" d="M35 230 L72 213 L105 224 L142 181 L175 198 L211 157 L247 170 L285 138 L322 155 L360 112 L396 126 L435 91 L474 113 L512 74 L550 88 L589 50 L625 67 L665 37" fill="none" stroke="#65e2d1" stroke-width="3.3" stroke-linejoin="round" stroke-linecap="round"/>
            <path class="k-line2" d="M35 238 L72 230 L105 204 L142 216 L175 183 L211 191 L247 159 L285 176 L322 142 L360 154 L396 140 L435 124 L474 137 L512 104 L550 119 L589 94 L625 102 L665 80" fill="none" stroke="#8f8bff" stroke-width="2" opacity=".85" stroke-dasharray="6 6"/>
            <circle class="k-dot" cx="665" cy="37" r="6" fill="#65e2d1"/>
            <text x="40" y="282" fill="#7185a4" font-size="12">μ = wᵀr</text>
            <text x="270" y="282" fill="#7185a4" font-size="12">σₚ = √(wᵀΣw)</text>
            <text x="560" y="282" fill="#7185a4" font-size="12">E[Rₚ]</text>
          </svg>
        </div>
        <div class="k-formula" style="margin-top:18px;color:#8194b4;font-size:.85rem;letter-spacing:.04em">
          max Sharpe(w) = (wᵀμ − r<sub>f</sub>) / √(wᵀΣw)
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="k-mini">◉ &nbsp; PRIVATE RESEARCH ENVIRONMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="k-panelhead">Accesso riservato</div><p class="k-muted">Inserisci la password per entrare in Quant Portfolio Optimizer.</p>', unsafe_allow_html=True)
        with st.form("kangemi_login", clear_on_submit=False, border=True):
            password = st.text_input("Password", type="password", placeholder="Inserisci la password", autocomplete="current-password")
            submitted = st.form_submit_button("Accedi al framework →", use_container_width=True)
            if submitted:
                if _check_password(password):
                    st.session_state["kangemi_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Password non corretta.")
        st.markdown('<p class="k-muted" style="font-size:.76rem">Accesso consentito esclusivamente agli utenti autorizzati. Le credenziali non sono memorizzate nel codice sorgente.</p>', unsafe_allow_html=True)

    st.markdown('<div class="k-footer">KANGEMI EDU — QUANTITATIVE BOUTIQUE &nbsp;·&nbsp; QUANT PORTFOLIO OPTIMIZER</div>', unsafe_allow_html=True)


if not _authenticated():
    _login_screen()
    st.stop()

if not CORE_APP.is_file():
    st.error("File app_core.py non trovato. Rinomina il precedente app.py in app_core.py e riprova.")
    st.stop()

# Run the original application on every Streamlit rerun (ordinary imports would be cached).
# Keep __file__ pointing at app_core.py so its existing relative paths keep working.
_original_code = compile(CORE_APP.read_bytes(), str(CORE_APP), "exec")
exec(_original_code, {"__name__": "__main__", "__file__": str(CORE_APP), "__package__": None})

with st.sidebar:
    st.divider()
    st.caption("Kangemi Edu · Accesso autorizzato")
    if st.button("🔒 Esci", key="kangemi_logout"):
        st.session_state.pop("kangemi_authenticated", None)
        st.rerun()
