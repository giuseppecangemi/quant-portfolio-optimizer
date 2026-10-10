"""Kangemi Edu – Quantitative Boutique | Streamlit access gateway.

Rename the existing app.py to app_core.py and put this file alongside it.
"""

from __future__ import annotations

import hmac
import re
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
    # Ambient animations use only SVG/CSS. They run independently of Streamlit reruns.
    st.markdown(re.sub(r">\s+<", "><", """
    <style>
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none!important}
      [data-testid="stHeader"] {background:transparent!important}
      [data-testid="stAppViewContainer"] {background:#070c17!important;overflow:hidden}
      .block-container {max-width:1440px!important;padding:2.4rem 4.5vw 2rem!important;position:relative;z-index:3}
      .k-world {position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;
        background:radial-gradient(ellipse at 17% 25%,rgba(97,74,191,.17),transparent 44%),
                   radial-gradient(ellipse at 88% 74%,rgba(20,164,171,.12),transparent 45%),#070c17}
      .k-world:after {content:"";position:absolute;inset:0;
        background:linear-gradient(90deg,rgba(7,12,23,.02),rgba(7,12,23,.37) 53%,rgba(7,12,23,.04));
        pointer-events:none}
      .k-world:before {content:"";position:absolute;inset:12% 22% 12% 22%;
        background:radial-gradient(ellipse,rgba(7,12,23,.55),rgba(7,12,23,0) 70%);
        pointer-events:none;z-index:1}
      .k-grid {position:absolute;inset:0;opacity:.24;
        background-image:linear-gradient(#2b3b5a 1px,transparent 1px),linear-gradient(90deg,#2b3b5a 1px,transparent 1px);
        background-size:74px 74px;mask-image:linear-gradient(to bottom,transparent,#000 23%,#000 84%,transparent)}
      .k-stage {position:absolute;inset:0;width:100%;height:100%;overflow:visible}
      .k-stage polyline {stroke-linejoin:round;stroke-linecap:round}
      .k-stage .plot-a {stroke-dasharray:650;stroke-dashoffset:650;animation:k-trace 7s ease-in-out infinite}
      .k-stage .plot-b {stroke-dasharray:650;stroke-dashoffset:650;animation:k-trace 9s ease-in-out infinite -5s}
      .k-stage .plot-c {stroke-dasharray:650;stroke-dashoffset:650;animation:k-trace 11s ease-in-out infinite -8s}
      .k-stage .plot-d {stroke-dasharray:650;stroke-dashoffset:650;animation:k-trace 8s ease-in-out infinite -3s}
      .k-stage .formula {font-family:Georgia,serif;fill:#a7b9ff;opacity:.48;
        animation:k-orbit 11s ease-in-out infinite alternate;animation-delay:var(--delay)}
      .k-stage .formula.teal {fill:#6ce5d9}
      .k-stage .cluster {animation:k-cluster 16s ease-in-out infinite alternate;transform-origin:center}
      .k-stage .cluster.reverse {animation-direction:alternate-reverse;animation-duration:19s}
      .k-stage .spark {animation:k-spark 2.1s ease-in-out infinite;animation-delay:var(--delay)}
      .k-stage .halo {animation:k-halo 5s ease-in-out infinite}
      .k-stage .bar {transform-box:fill-box;transform-origin:bottom;
        animation:k-bars 2.3s ease-in-out infinite;animation-delay:var(--delay)}
      .k-stage .particle {animation:k-particle 8s linear infinite;animation-delay:var(--delay)}
      .k-stage .ring {stroke-dasharray:9 15;animation:k-rotate 17s linear infinite;transform-origin:center;transform-box:fill-box}
      .k-stage .ring.rev {animation-direction:reverse}
      @keyframes k-trace {0%{stroke-dashoffset:650;opacity:0}20%{opacity:.8}57%{stroke-dashoffset:0;opacity:.9}85%{stroke-dashoffset:0;opacity:.7}100%{stroke-dashoffset:-650;opacity:0}}
      @keyframes k-orbit {0%{transform:translate(-22px,16px) rotate(-3deg);opacity:.14}50%{opacity:.55}100%{transform:translate(36px,-30px) rotate(3deg);opacity:.28}}
      @keyframes k-cluster {0%{transform:translate(-38px,18px)}50%{transform:translate(38px,-22px)}100%{transform:translate(-15px,36px)}}
      @keyframes k-spark {0%,100%{opacity:.25;r:2}50%{opacity:.95;r:4}}
      @keyframes k-halo {0%,100%{opacity:.18;transform:scale(.85)}50%{opacity:.55;transform:scale(1.18)}}
      @keyframes k-bars {0%,100%{transform:scaleY(.25);opacity:.2}50%{transform:scaleY(1);opacity:.66}}
      @keyframes k-particle {0%{transform:translate(0,0);opacity:0}15%{opacity:.8}85%{opacity:.6}100%{transform:translate(110px,-160px);opacity:0}}
      @keyframes k-rotate {to{transform:rotate(360deg)}}

      .k-fin-viz {animation:k-fin-drift 11s ease-in-out infinite alternate;transform-box:fill-box;transform-origin:center}
      .k-fin-viz:nth-of-type(2n) {animation-delay:-9s;animation-duration:26s}
      .k-fin-path {stroke-dasharray:1100;stroke-dashoffset:1100;animation:k-fin-draw var(--dur,15s) ease-in-out infinite;animation-delay:var(--lag,0s)}
      .k-cml {animation:k-cml-glow 2.6s ease-in-out infinite}
      .k-tangent {animation:k-point-pulse 1.8s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
      .k-matrix-cell {animation:k-matrix-pulse 3s ease-in-out infinite;animation-delay:var(--lag)}
      .k-vol-bar {transform-box:fill-box;transform-origin:bottom;animation:k-bars 2.7s ease-in-out infinite;animation-delay:var(--lag)}
      @keyframes k-fin-drift {0%{transform:translate(-9px,7px) rotate(-1.2deg)}50%{transform:translate(9px,-7px) rotate(1.1deg)}100%{transform:translate(-5px,-10px) rotate(-.7deg)}}
      @keyframes k-fin-draw {0%{stroke-dashoffset:1100;opacity:.08}25%{stroke-dashoffset:0;opacity:.9}60%{stroke-dashoffset:0;opacity:1}100%{stroke-dashoffset:-1100;opacity:.12}}
      @keyframes k-cml-glow {50%{stroke:#d4c9ff;opacity:1}}
      @keyframes k-point-pulse {50%{transform:scale(1.7);opacity:.5}}
      @keyframes k-matrix-pulse {50%{opacity:.16}}
      .k-brand {font-size:clamp(1.25rem,2.2vw,2.05rem);letter-spacing:.10em;font-weight:850;color:#dce0ff;text-transform:uppercase;line-height:1.3;text-shadow:0 2px 22px rgba(103,115,227,.28)}
      .k-brand span {color:#9da9c9;font-weight:650}
      .k-eyebrow {font-size:.73rem;letter-spacing:.18em;color:#71d8d6;text-transform:uppercase;margin-top:10vh}
      .k-title {font-size:clamp(3.2rem,6.2vw,6.5rem);line-height:1.05;font-weight:780;letter-spacing:-.055em;
        color:#f3f6ff;margin:.9rem 0 1.25rem;text-shadow:0 8px 38px #050916}
      .k-title em {font-style:normal;background:linear-gradient(95deg,#a59aff,#70dfdf);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent}
      .k-sub {color:#b6c3da;font-size:1.03rem;line-height:1.85;max-width:610px;text-shadow:0 2px 18px #050916}
      .k-chips {display:flex;gap:10px;flex-wrap:wrap;margin:1.8rem 0}
      .k-chip {border:1px solid rgba(115,148,201,.35);background:rgba(13,25,46,.68);color:#c3d3e7;
        padding:8px 13px;border-radius:40px;font-size:.72rem;letter-spacing:.04em;backdrop-filter:blur(9px)}
      .k-mini {font-size:.7rem;letter-spacing:.15em;color:#72dbca}
      .k-panelhead {color:#fff;font-size:1.7rem;font-weight:730;margin:.6rem 0}
      .k-muted {color:#b2c0d8;font-size:.88rem;line-height:1.65}
      .k-form-wrap {background:rgba(9,17,32,.55);padding:1.7rem;border:1px solid rgba(133,157,203,.22);
        border-radius:24px;backdrop-filter:blur(22px);box-shadow:0 30px 95px rgba(0,0,0,.28)}
      .stForm {background:rgba(13,21,38,.78);border:1px solid rgba(133,157,203,.22);
        border-radius:17px;padding:1.3rem;box-shadow:0 12px 55px rgba(0,0,0,.2)}
      .stForm label {color:#bdcce5!important}
      .stForm input {background:#0a1326!important;color:#fff!important;border:1px solid #344461!important}
      .stForm button {width:100%;background:linear-gradient(95deg,#6659d8,#318bba)!important;
        color:white!important;border:0!important;border-radius:10px!important;font-weight:650!important}
      .stForm button:hover {filter:brightness(1.14)}
      @media(max-width:800px){
        .block-container{padding:1.2rem 5vw 2rem!important}
        .k-eyebrow{margin-top:3rem}
        .k-title{font-size:clamp(2.9rem,11vw,4.8rem)}
        .k-world{opacity:.65}
      }
      @media(prefers-reduced-motion:reduce){
        .k-stage *, .k-hint:before {animation:none!important}
        .k-stage .plot-a,.k-stage .plot-b,.k-stage .plot-c,.k-stage .plot-d,.k-fin-path {stroke-dashoffset:0!important}
      }
    </style>
    <div class="k-world" aria-hidden="true">
      <div class="k-grid"></div>
      <svg class="k-stage" viewBox="0 0 1440 900" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="k-grad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stop-color="#66dfd5" stop-opacity=".19"/>
            <stop offset="1" stop-color="#66dfd5" stop-opacity="0"/>
          </linearGradient>
          <radialGradient id="k-glow"><stop offset="0" stop-color="#75e1d8" stop-opacity=".3"/>
            <stop offset="1" stop-color="#75e1d8" stop-opacity="0"/></radialGradient>
        </defs>
        <!-- Many independent micro-visualizations distributed across the viewport. -->
        <g transform="translate(20 95)" opacity=".32">
          <path d="M0 43.0H193" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,52.0 14.8,61.8 29.7,46.4 44.5,42.3 59.4,34.5 74.2,31.2 89.1,33.7 103.9,24.4 118.8,20.4 133.6,9.5 148.5,9.4 163.3,9.4 178.2,10.0 193.0,20.1" fill="none" stroke="#65dacf" stroke-width="1.6" class="plot-a"/>
          <text x="0" y="-10" fill="#65dacf" font-family="monospace" font-size="9" letter-spacing="1.3">RETURN PATH / 01</text>
        </g>
        <g transform="translate(290 70)" opacity=".32">
          <path d="M0 42.5H164" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,53.7 12.6,48.5 25.2,44.4 37.8,59.2 50.5,49.7 63.1,65.4 75.7,53.0 88.3,41.6 100.9,51.1 113.5,43.5 126.2,35.7 138.8,48.3 151.4,42.6 164.0,55.2" fill="none" stroke="#8e8bf8" stroke-width="1.6" class="plot-b"/>
          <text x="0" y="-10" fill="#8e8bf8" font-family="monospace" font-size="9" letter-spacing="1.3">RISK MODEL / 02</text>
        </g>
        <g transform="translate(565 35)" opacity=".32">
          <path d="M0 38.0H163" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,53.0 12.5,59.1 25.1,55.1 37.6,51.5 50.2,62.1 62.7,49.1 75.2,37.5 87.8,44.1 100.3,56.0 112.8,50.4 125.4,53.5 137.9,48.0 150.5,48.3 163.0,51.8" fill="none" stroke="#56a9e5" stroke-width="1.6" class="plot-c"/>
          <text x="0" y="-10" fill="#56a9e5" font-family="monospace" font-size="9" letter-spacing="1.3">FACTOR SIGNAL / 03</text>
        </g>
        <g transform="translate(830 75)" opacity=".32">
          <path d="M0 35.0H147" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,47.3 11.3,47.2 22.6,38.1 33.9,39.1 45.2,29.4 56.5,26.6 67.8,12.3 79.2,5.7 90.5,10.2 101.8,20.8 113.1,34.0 124.4,48.0 135.7,34.3 147.0,20.3" fill="none" stroke="#b8a2ff" stroke-width="1.6" class="plot-d"/>
          <text x="0" y="-10" fill="#b8a2ff" font-family="monospace" font-size="9" letter-spacing="1.3">SIMULATION / 04</text>
        </g>
        <g transform="translate(1120 95)" opacity=".32">
          <path d="M0 36.5H158" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,23.2 12.2,26.1 24.3,19.1 36.5,33.0 48.6,32.1 60.8,28.7 72.9,42.1 85.1,27.2 97.2,36.5 109.4,41.1 121.5,42.5 133.7,35.0 145.8,25.3 158.0,39.3" fill="none" stroke="#65dacf" stroke-width="1.6" class="plot-a"/>
          <text x="0" y="-10" fill="#65dacf" font-family="monospace" font-size="9" letter-spacing="1.3">VOLATILITY / 05</text>
        </g>
        <g transform="translate(60 320)" opacity=".32">
          <path d="M0 49.5H180" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,59.2 13.8,52.9 27.7,60.4 41.5,47.1 55.4,52.9 69.2,38.7 83.1,42.7 96.9,29.8 110.8,21.3 124.6,15.9 138.5,25.5 152.3,36.8 166.2,52.5 180.0,63.3" fill="none" stroke="#8e8bf8" stroke-width="1.6" class="plot-b"/>
          <text x="0" y="-10" fill="#8e8bf8" font-family="monospace" font-size="9" letter-spacing="1.3">RETURN PATH / 06</text>
        </g>
        <g transform="translate(335 335)" opacity=".32">
          <path d="M0 34.5H175" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,18.1 13.5,15.0 26.9,9.4 40.4,8.2 53.8,19.5 67.3,21.9 80.8,13.0 94.2,0.9 107.7,1.2 121.2,0.9 134.6,9.7 148.1,0.9 161.5,3.0 175.0,6.4" fill="none" stroke="#56a9e5" stroke-width="1.6" class="plot-c"/>
          <text x="0" y="-10" fill="#56a9e5" font-family="monospace" font-size="9" letter-spacing="1.3">RISK MODEL / 07</text>
        </g>
        <g transform="translate(640 310)" opacity=".32">
          <path d="M0 43.5H179" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,59.3 13.8,56.3 27.5,41.4 41.3,36.8 55.1,36.4 68.8,31.3 82.6,42.2 96.4,34.5 110.2,25.0 123.9,39.4 137.7,39.1 151.5,35.9 165.2,40.8 179.0,34.7" fill="none" stroke="#b8a2ff" stroke-width="1.6" class="plot-d"/>
          <text x="0" y="-10" fill="#b8a2ff" font-family="monospace" font-size="9" letter-spacing="1.3">FACTOR SIGNAL / 08</text>
        </g>
        <g transform="translate(975 330)" opacity=".32">
          <path d="M0 41.5H184" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,53.2 14.2,41.3 28.3,37.5 42.5,26.5 56.6,14.1 70.8,21.5 84.9,18.6 99.1,25.2 113.2,15.2 127.4,8.7 141.5,17.5 155.7,17.5 169.8,14.3 184.0,23.3" fill="none" stroke="#65dacf" stroke-width="1.6" class="plot-a"/>
          <text x="0" y="-10" fill="#65dacf" font-family="monospace" font-size="9" letter-spacing="1.3">SIMULATION / 09</text>
        </g>
        <g transform="translate(1220 340)" opacity=".32">
          <path d="M0 47.5H168" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,54.5 12.9,61.5 25.8,61.0 38.8,57.2 51.7,60.0 64.6,54.8 77.5,36.5 90.5,24.4 103.4,21.8 116.3,13.9 129.2,13.9 142.2,15.4 155.1,13.9 168.0,13.9" fill="none" stroke="#8e8bf8" stroke-width="1.6" class="plot-b"/>
          <text x="0" y="-10" fill="#8e8bf8" font-family="monospace" font-size="9" letter-spacing="1.3">VOLATILITY / 10</text>
        </g>
        <g transform="translate(25 555)" opacity=".32">
          <path d="M0 38.0H219" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,40.4 16.8,56.7 33.7,67.4 50.5,63.5 67.4,66.9 84.2,67.4 101.1,64.4 117.9,67.4 134.8,67.4 151.6,55.7 168.5,51.6 185.3,49.7 202.2,48.8 219.0,57.3" fill="none" stroke="#56a9e5" stroke-width="1.6" class="plot-c"/>
          <text x="0" y="-10" fill="#56a9e5" font-family="monospace" font-size="9" letter-spacing="1.3">RETURN PATH / 11</text>
        </g>
        <g transform="translate(310 575)" opacity=".32">
          <path d="M0 33.5H168" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,32.6 12.9,29.7 25.8,23.3 38.8,23.7 51.7,21.4 64.6,14.0 77.5,24.4 90.5,35.1 103.4,35.7 116.3,27.1 129.2,16.9 142.2,5.1 155.1,7.1 168.0,-0.1" fill="none" stroke="#b8a2ff" stroke-width="1.6" class="plot-d"/>
          <text x="0" y="-10" fill="#b8a2ff" font-family="monospace" font-size="9" letter-spacing="1.3">RISK MODEL / 12</text>
        </g>
        <g transform="translate(605 570)" opacity=".32">
          <path d="M0 42.0H158" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,45.7 12.2,58.8 24.3,46.1 36.5,27.8 48.6,20.7 60.8,10.7 72.9,13.5 85.1,8.4 97.2,20.9 109.4,23.8 121.5,27.3 133.7,13.6 145.8,23.1 158.0,33.1" fill="none" stroke="#65dacf" stroke-width="1.6" class="plot-a"/>
          <text x="0" y="-10" fill="#65dacf" font-family="monospace" font-size="9" letter-spacing="1.3">FACTOR SIGNAL / 13</text>
        </g>
        <g transform="translate(910 575)" opacity=".32">
          <path d="M0 34.5H156" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,23.0 12.0,29.2 24.0,15.6 36.0,21.9 48.0,10.6 60.0,20.9 72.0,7.3 84.0,0.9 96.0,0.9 108.0,0.9 120.0,0.9 132.0,0.9 144.0,0.9 156.0,8.4" fill="none" stroke="#8e8bf8" stroke-width="1.6" class="plot-b"/>
          <text x="0" y="-10" fill="#8e8bf8" font-family="monospace" font-size="9" letter-spacing="1.3">SIMULATION / 14</text>
        </g>
        <g transform="translate(1180 565)" opacity=".32">
          <path d="M0 43.5H180" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,27.4 13.8,18.9 27.7,9.9 41.5,9.9 55.4,9.9 69.2,13.1 83.1,9.9 96.9,9.9 110.8,9.9 124.6,9.9 138.5,9.9 152.3,21.3 166.2,9.9 180.0,9.9" fill="none" stroke="#56a9e5" stroke-width="1.6" class="plot-c"/>
          <text x="0" y="-10" fill="#56a9e5" font-family="monospace" font-size="9" letter-spacing="1.3">VOLATILITY / 15</text>
        </g>
        <g transform="translate(95 750)" opacity=".32">
          <path d="M0 41.0H148" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,40.2 11.4,49.1 22.8,42.1 34.2,47.9 45.5,54.8 56.9,41.5 68.3,47.7 79.7,60.0 91.1,46.9 102.5,28.8 113.8,33.9 125.2,35.6 136.6,32.6 148.0,22.6" fill="none" stroke="#b8a2ff" stroke-width="1.6" class="plot-d"/>
          <text x="0" y="-10" fill="#b8a2ff" font-family="monospace" font-size="9" letter-spacing="1.3">RETURN PATH / 16</text>
        </g>
        <g transform="translate(410 760)" opacity=".32">
          <path d="M0 44.0H146" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,39.0 11.2,48.0 22.5,33.6 33.7,36.4 44.9,21.5 56.2,23.2 67.4,18.0 78.6,33.5 89.8,43.5 101.1,50.3 112.3,54.7 123.5,65.7 134.8,68.4 146.0,73.4" fill="none" stroke="#65dacf" stroke-width="1.6" class="plot-a"/>
          <text x="0" y="-10" fill="#65dacf" font-family="monospace" font-size="9" letter-spacing="1.3">RISK MODEL / 17</text>
        </g>
        <g transform="translate(710 760)" opacity=".32">
          <path d="M0 50.0H217" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,34.4 16.7,22.4 33.4,18.3 50.1,16.4 66.8,16.4 83.5,16.4 100.2,17.8 116.8,31.9 133.5,18.7 150.2,16.4 166.9,16.4 183.6,16.4 200.3,16.4 217.0,16.4" fill="none" stroke="#8e8bf8" stroke-width="1.6" class="plot-b"/>
          <text x="0" y="-10" fill="#8e8bf8" font-family="monospace" font-size="9" letter-spacing="1.3">FACTOR SIGNAL / 18</text>
        </g>
        <g transform="translate(1030 760)" opacity=".32">
          <path d="M0 35.0H147" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,20.0 11.3,16.2 22.6,16.4 33.9,17.2 45.2,23.5 56.5,37.6 67.8,32.2 79.2,26.8 90.5,39.2 101.8,23.5 113.1,8.5 124.4,1.4 135.7,1.4 147.0,3.8" fill="none" stroke="#56a9e5" stroke-width="1.6" class="plot-c"/>
          <text x="0" y="-10" fill="#56a9e5" font-family="monospace" font-size="9" letter-spacing="1.3">SIMULATION / 19</text>
        </g>
        <g transform="translate(1270 765)" opacity=".32">
          <path d="M0 38.5H205" stroke="#49617e" stroke-width=".65" stroke-dasharray="3 8"/>
          <polyline points="0.0,42.8 15.8,27.5 31.5,13.5 47.3,4.9 63.1,4.9 78.8,4.9 94.6,4.9 110.4,12.0 126.2,22.6 141.9,6.3 157.7,14.0 173.5,21.8 189.2,22.0 205.0,35.9" fill="none" stroke="#b8a2ff" stroke-width="1.6" class="plot-d"/>
          <text x="0" y="-10" fill="#b8a2ff" font-family="monospace" font-size="9" letter-spacing="1.3">VOLATILITY / 20</text>
        </g>
        <text class="formula " style="--delay:-1s" x="100" y="55" font-size="18">σₚ = √(wᵀΣw)</text>
        <text class="formula teal" style="--delay:-3s" x="1060" y="65" font-size="15">E[Rₚ] = wᵀμ</text>
        <text class="formula " style="--delay:-5s" x="765" y="235" font-size="15">βᵢ = Cov(Rᵢ,Rₘ) / Var(Rₘ)</text>
        <text class="formula teal" style="--delay:-7s" x="85" y="495" font-size="14">SR = (μₚ − rᶠ) / σₚ</text>
        <text class="formula " style="--delay:-9s" x="1200" y="510" font-size="19">min wᵀΣw</text>
        <text class="formula teal" style="--delay:-11s" x="600" y="840" font-size="14">E[Rᵢ] = rᶠ + βᵢ(E[Rₘ] − rᶠ)</text>
        <text class="formula " style="--delay:-13s" x="260" y="225" font-size="14">α + β₁MKT + β₂SMB + β₃HML</text>
        <text class="formula teal" style="--delay:-15s" x="1060" y="260" font-size="16">VaRₐ = inf{x : F(x) ≥ α}</text>
        <text class="formula " style="--delay:-17s" x="100" y="865" font-size="16">Σ = DρD</text>
        <text class="formula teal" style="--delay:-19s" x="1180" y="865" font-size="15">RCᵢ = wᵢ(Σw)ᵢ / σₚ</text>
        <text class="formula " style="--delay:-21s" x="610" y="60" font-size="15">GARCH(1,1)</text>
        <text class="formula teal" style="--delay:-23s" x="810" y="675" font-size="15">μ = 252 · E[r]</text>
        <circle class="particle" style="--delay:-0s" cx="517" cy="151" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-1s" cx="719" cy="597" r="1" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-2s" cx="1011" cy="83" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-3s" cx="149" cy="568" r="2" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-4s" cx="228" cy="800" r="1.4" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-5s" cx="1326" cy="151" r="1" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-6s" cx="410" cy="286" r="1.4" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-7s" cx="497" cy="77" r="1" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-8s" cx="612" cy="85" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-9s" cx="139" cy="190" r="2" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-10s" cx="937" cy="247" r="2" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-11s" cx="435" cy="257" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-12s" cx="447" cy="298" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-0s" cx="712" cy="521" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-1s" cx="1018" cy="358" r="1.4" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-2s" cx="982" cy="354" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-3s" cx="1278" cy="782" r="1.4" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-4s" cx="303" cy="97" r="1" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-5s" cx="915" cy="155" r="1.4" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-6s" cx="887" cy="486" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-7s" cx="314" cy="494" r="1.4" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-8s" cx="472" cy="90" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-9s" cx="168" cy="114" r="2" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-10s" cx="593" cy="294" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-11s" cx="491" cy="341" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-12s" cx="12" cy="527" r="2" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-0s" cx="1269" cy="539" r="1.4" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-1s" cx="1372" cy="736" r="2" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-2s" cx="871" cy="70" r="2" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-3s" cx="1432" cy="515" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-4s" cx="469" cy="758" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-5s" cx="1255" cy="439" r="1" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-6s" cx="831" cy="655" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-7s" cx="1127" cy="96" r="1" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-8s" cx="903" cy="608" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-9s" cx="398" cy="418" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-10s" cx="816" cy="241" r="2" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-11s" cx="851" cy="472" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-12s" cx="567" cy="406" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-0s" cx="887" cy="771" r="2" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-1s" cx="621" cy="205" r="1.4" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-2s" cx="379" cy="866" r="2" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-3s" cx="1233" cy="645" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-4s" cx="69" cy="801" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-5s" cx="1178" cy="855" r="1.4" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-6s" cx="1438" cy="58" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-7s" cx="101" cy="553" r="2" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-8s" cx="961" cy="814" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-9s" cx="571" cy="171" r="2" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-10s" cx="1094" cy="454" r="1" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-11s" cx="915" cy="441" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-12s" cx="487" cy="827" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-0s" cx="1034" cy="778" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-1s" cx="185" cy="197" r="1" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-2s" cx="630" cy="52" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-3s" cx="1371" cy="365" r="2" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-4s" cx="1392" cy="634" r="2" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-5s" cx="64" cy="608" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-6s" cx="947" cy="344" r="1.4" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-7s" cx="662" cy="380" r="1" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-8s" cx="143" cy="584" r="1" fill="#65dacf" opacity=".3"/>
        <circle class="particle" style="--delay:-9s" cx="49" cy="10" r="1.4" fill="#8e8bf8" opacity=".3"/>
        <circle class="particle" style="--delay:-10s" cx="1117" cy="411" r="1" fill="#56a9e5" opacity=".3"/>
        <circle class="particle" style="--delay:-11s" cx="172" cy="723" r="1.4" fill="#b8a2ff" opacity=".3"/>
        <circle class="particle" style="--delay:-12s" cx="260" cy="6" r="2" fill="#65dacf" opacity=".3"/>
        <circle class="ring" cx="1200" cy="420" r="52" fill="none" stroke="#7bd5d8" stroke-width=".9" opacity=".23"/>
        <circle class="ring" cx="170" cy="685" r="42" fill="none" stroke="#7bd5d8" stroke-width=".9" opacity=".23"/>
        <circle class="ring" cx="730" cy="460" r="33" fill="none" stroke="#7bd5d8" stroke-width=".9" opacity=".23"/>
        <circle class="ring" cx="1380" cy="190" r="38" fill="none" stroke="#7bd5d8" stroke-width=".9" opacity=".23"/>
<g transform="translate(38 75)" opacity=".68"><g class="k-fin-viz">
<rect x="-10" y="-21" width="290" height="180" rx="12" fill="#0a1425" fill-opacity=".25" stroke="#456486" stroke-opacity=".2"/>
<text x="0" y="-6" fill="#a5c9e4" font-family="monospace" font-size="10" letter-spacing="1.4">MONTE CARLO / STOCHASTIC PATHS</text>
<path d="M0 150H270 M0 0V150" fill="none" stroke="#526781" stroke-opacity=".55" stroke-width=".8"/>
<polyline points="0.0,85.0 9.0,81.3 18.0,76.6 27.0,73.7 36.0,77.4 45.0,65.6 54.0,62.6 63.0,58.6 72.0,52.1 81.0,50.5 90.0,49.7 99.0,64.7 108.0,65.7 117.0,64.9 126.0,69.6 135.0,62.0 144.0,57.3 153.0,60.0 162.0,55.9 171.0,48.8 180.0,44.2 189.0,34.7 198.0,41.2 207.0,48.5 216.0,64.4 225.0,61.7 234.0,59.4 243.0,57.3 252.0,59.9 261.0,59.1 270.0,63.3" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-0.00s;--dur:6.8s"/>
<polyline points="0.0,89.2 9.0,79.2 18.0,88.6 27.0,84.5 36.0,91.1 45.0,88.1 54.0,88.2 63.0,76.7 72.0,69.6 81.0,71.3 90.0,74.2 99.0,71.9 108.0,71.9 117.0,59.3 126.0,58.7 135.0,53.5 144.0,46.0 153.0,47.4 162.0,43.7 171.0,37.2 180.0,33.8 189.0,30.4 198.0,27.7 207.0,27.3 216.0,23.6 225.0,28.6 234.0,37.8 243.0,43.4 252.0,40.3 261.0,35.4 270.0,37.0" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-1.15s;--dur:8.1s"/>
<polyline points="0.0,89.8 9.0,97.7 18.0,98.2 27.0,100.3 36.0,95.8 45.0,96.0 54.0,88.1 63.0,77.7 72.0,78.1 81.0,70.0 90.0,65.2 99.0,70.0 108.0,79.2 117.0,78.7 126.0,75.1 135.0,79.1 144.0,84.4 153.0,86.9 162.0,80.5 171.0,76.5 180.0,82.2 189.0,74.3 198.0,73.9 207.0,80.9 216.0,86.6 225.0,81.6 234.0,78.9 243.0,76.7 252.0,68.7 261.0,68.3 270.0,62.0" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-2.30s;--dur:9.3s"/>
<polyline points="0.0,95.0 9.0,87.3 18.0,82.4 27.0,77.7 36.0,77.1 45.0,60.6 54.0,65.8 63.0,63.5 72.0,63.5 81.0,70.6 90.0,69.8 99.0,71.7 108.0,71.7 117.0,76.3 126.0,79.9 135.0,86.4 144.0,85.2 153.0,77.8 162.0,78.7 171.0,70.9 180.0,70.0 189.0,67.1 198.0,52.7 207.0,49.7 216.0,42.5 225.0,33.4 234.0,21.2 243.0,23.0 252.0,26.0 261.0,18.0 270.0,10.6" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-3.45s;--dur:10.5s"/>
<polyline points="0.0,85.8 9.0,78.3 18.0,81.0 27.0,73.7 36.0,78.6 45.0,62.5 54.0,50.0 63.0,54.1 72.0,50.6 81.0,36.6 90.0,49.3 99.0,52.5 108.0,53.4 117.0,36.3 126.0,40.6 135.0,39.2 144.0,36.6 153.0,32.7 162.0,19.9 171.0,11.0 180.0,12.3 189.0,17.0 198.0,19.8 207.0,21.0 216.0,20.9 225.0,22.8 234.0,35.3 243.0,34.5 252.0,23.3 261.0,24.9 270.0,21.4" fill="none" stroke="#65cdb7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-4.60s;--dur:11.8s"/>
<polyline points="0.0,83.8 9.0,74.6 18.0,78.1 27.0,72.5 36.0,70.0 45.0,63.2 54.0,69.1 63.0,65.2 72.0,61.2 81.0,50.3 90.0,41.1 99.0,46.7 108.0,39.1 117.0,44.4 126.0,46.4 135.0,47.4 144.0,47.7 153.0,46.6 162.0,52.8 171.0,52.5 180.0,42.6 189.0,39.8 198.0,48.6 207.0,40.1 216.0,39.0 225.0,37.5 234.0,43.6 243.0,37.1 252.0,44.9 261.0,37.2 270.0,37.1" fill="none" stroke="#c4a7ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-5.75s;--dur:6.8s"/>
<polyline points="0.0,83.9 9.0,84.1 18.0,79.5 27.0,69.1 36.0,70.4 45.0,63.3 54.0,66.9 63.0,50.2 72.0,59.5 81.0,58.2 90.0,68.7 99.0,72.8 108.0,72.6 117.0,70.7 126.0,67.2 135.0,71.3 144.0,70.0 153.0,60.3 162.0,58.7 171.0,61.8 180.0,62.4 189.0,64.1 198.0,67.0 207.0,51.6 216.0,52.8 225.0,62.3 234.0,65.4 243.0,63.3 252.0,56.3 261.0,52.4 270.0,64.3" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-6.90s;--dur:8.1s"/>
<polyline points="0.0,89.1 9.0,92.5 18.0,92.1 27.0,96.1 36.0,88.4 45.0,76.0 54.0,73.5 63.0,74.1 72.0,57.4 81.0,55.6 90.0,56.5 99.0,54.9 108.0,52.2 117.0,39.7 126.0,33.8 135.0,26.0 144.0,30.4 153.0,31.3 162.0,33.8 171.0,22.8 180.0,14.1 189.0,14.8 198.0,14.0 207.0,19.9 216.0,15.4 225.0,10.8 234.0,15.1 243.0,9.1 252.0,7.0 261.0,12.4 270.0,21.7" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-8.05s;--dur:9.3s"/>
<polyline points="0.0,88.8 9.0,79.5 18.0,75.4 27.0,56.1 36.0,60.7 45.0,70.5 54.0,72.2 63.0,66.1 72.0,58.6 81.0,55.2 90.0,56.7 99.0,62.3 108.0,69.9 117.0,81.2 126.0,83.9 135.0,91.4 144.0,74.8 153.0,66.6 162.0,67.4 171.0,60.7 180.0,58.4 189.0,59.0 198.0,63.4 207.0,64.3 216.0,65.4 225.0,63.9 234.0,67.1 243.0,74.4 252.0,70.9 261.0,73.2 270.0,74.2" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-9.20s;--dur:10.5s"/>
<polyline points="0.0,90.5 9.0,84.2 18.0,79.4 27.0,76.2 36.0,78.8 45.0,81.3 54.0,86.1 63.0,85.3 72.0,77.3 81.0,78.7 90.0,64.9 99.0,70.1 108.0,81.0 117.0,74.3 126.0,70.2 135.0,55.9 144.0,53.9 153.0,50.9 162.0,51.8 171.0,54.1 180.0,52.4 189.0,59.6 198.0,65.1 207.0,62.3 216.0,66.3 225.0,57.5 234.0,45.8 243.0,37.0 252.0,47.7 261.0,40.5 270.0,33.6" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-10.35s;--dur:11.8s"/>
<polyline points="0.0,90.2 9.0,89.1 18.0,93.5 27.0,100.7 36.0,98.9 45.0,95.3 54.0,89.7 63.0,83.4 72.0,76.8 81.0,73.0 90.0,73.3 99.0,78.3 108.0,81.5 117.0,79.9 126.0,84.1 135.0,77.3 144.0,86.9 153.0,91.3 162.0,102.6 171.0,103.1 180.0,94.6 189.0,106.0 198.0,104.3 207.0,97.4 216.0,83.3 225.0,69.1 234.0,70.3 243.0,73.3 252.0,69.6 261.0,80.2 270.0,79.5" fill="none" stroke="#65cdb7" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-11.50s;--dur:6.8s"/>
<polyline points="0.0,96.0 9.0,88.7 18.0,82.3 27.0,74.4 36.0,77.5 45.0,77.3 54.0,76.6 63.0,77.8 72.0,77.4 81.0,78.8 90.0,60.3 99.0,63.9 108.0,67.5 117.0,73.5 126.0,73.3 135.0,67.7 144.0,59.2 153.0,65.0 162.0,59.5 171.0,64.5 180.0,65.7 189.0,59.2 198.0,59.8 207.0,67.8 216.0,68.8 225.0,58.0 234.0,57.8 243.0,55.4 252.0,56.3 261.0,55.1 270.0,58.2" fill="none" stroke="#c4a7ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-12.65s;--dur:8.1s"/>
<polyline points="0.0,87.6 9.0,87.6 18.0,93.6 27.0,96.5 36.0,97.5 45.0,99.0 54.0,105.5 63.0,111.5 72.0,114.7 81.0,102.9 90.0,94.0 99.0,89.0 108.0,88.3 117.0,80.7 126.0,69.7 135.0,68.2 144.0,61.6 153.0,54.2 162.0,48.5 171.0,41.4 180.0,41.7 189.0,46.2 198.0,43.0 207.0,35.9 216.0,38.9 225.0,33.3 234.0,30.9 243.0,27.3 252.0,22.6 261.0,33.6 270.0,36.8" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-13.80s;--dur:9.3s"/>
<polyline points="0.0,99.2 9.0,107.0 18.0,107.6 27.0,93.7 36.0,97.0 45.0,101.1 54.0,90.5 63.0,84.6 72.0,91.8 81.0,95.2 90.0,100.4 99.0,94.1 108.0,97.8 117.0,96.3 126.0,99.8 135.0,89.7 144.0,87.7 153.0,72.3 162.0,65.3 171.0,61.1 180.0,59.0 189.0,54.6 198.0,49.4 207.0,51.7 216.0,49.6 225.0,39.5 234.0,41.1 243.0,47.9 252.0,45.6 261.0,36.7 270.0,43.4" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-14.95s;--dur:10.5s"/>
<polyline points="0.0,94.4 9.0,79.8 18.0,78.3 27.0,74.2 36.0,88.1 45.0,86.5 54.0,77.0 63.0,79.8 72.0,86.2 81.0,73.0 90.0,71.7 99.0,68.1 108.0,72.8 117.0,74.5 126.0,68.3 135.0,69.0 144.0,68.1 153.0,56.4 162.0,47.5 171.0,52.5 180.0,57.2 189.0,71.1 198.0,70.5 207.0,73.3 216.0,63.1 225.0,70.8 234.0,62.0 243.0,55.5 252.0,57.3 261.0,50.7 270.0,68.2" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-16.10s;--dur:11.8s"/>
<polyline points="0.0,86.4 9.0,75.2 18.0,72.0 27.0,76.3 36.0,68.7 45.0,74.2 54.0,80.2 63.0,67.3 72.0,69.9 81.0,69.3 90.0,69.4 99.0,70.2 108.0,72.9 117.0,68.8 126.0,70.8 135.0,65.7 144.0,72.2 153.0,75.5 162.0,80.7 171.0,79.5 180.0,79.2 189.0,74.8 198.0,67.3 207.0,68.6 216.0,54.8 225.0,63.6 234.0,65.8 243.0,52.3 252.0,54.8 261.0,63.1 270.0,70.3" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-17.25s;--dur:6.8s"/>
<polyline points="0.0,88.6 9.0,89.0 18.0,75.4 27.0,77.6 36.0,82.3 45.0,84.3 54.0,84.1 63.0,83.6 72.0,86.5 81.0,78.5 90.0,65.3 99.0,58.5 108.0,70.9 117.0,77.9 126.0,77.3 135.0,63.9 144.0,65.6 153.0,58.9 162.0,80.0 171.0,82.4 180.0,74.8 189.0,66.4 198.0,63.8 207.0,57.8 216.0,53.1 225.0,46.4 234.0,46.7 243.0,53.3 252.0,47.1 261.0,47.4 270.0,46.5" fill="none" stroke="#65cdb7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-18.40s;--dur:8.1s"/>
<polyline points="0.0,77.1 9.0,75.2 18.0,73.5 27.0,64.0 36.0,60.6 45.0,62.0 54.0,56.1 63.0,55.9 72.0,61.2 81.0,66.3 90.0,74.8 99.0,82.1 108.0,82.7 117.0,82.6 126.0,80.4 135.0,89.7 144.0,105.9 153.0,101.2 162.0,110.6 171.0,116.8 180.0,117.2 189.0,121.0 198.0,121.4 207.0,130.2 216.0,130.4 225.0,123.8 234.0,124.6 243.0,115.3 252.0,119.1 261.0,118.0 270.0,120.9" fill="none" stroke="#c4a7ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-19.55s;--dur:9.3s"/>
<polyline points="0.0,100.2 9.0,101.7 18.0,98.5 27.0,93.5 36.0,75.9 45.0,73.4 54.0,61.8 63.0,62.8 72.0,57.1 81.0,62.8 90.0,63.7 99.0,53.2 108.0,47.9 117.0,44.0 126.0,50.0 135.0,51.7 144.0,49.9 153.0,43.1 162.0,41.1 171.0,33.2 180.0,39.9 189.0,42.7 198.0,36.5 207.0,33.9 216.0,38.9 225.0,55.7 234.0,46.1 243.0,39.9 252.0,49.1 261.0,51.0 270.0,54.0" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-20.70s;--dur:10.5s"/>
<polyline points="0.0,97.4 9.0,91.6 18.0,94.6 27.0,94.9 36.0,89.9 45.0,100.7 54.0,95.7 63.0,81.8 72.0,81.6 81.0,67.1 90.0,61.9 99.0,43.4 108.0,43.9 117.0,42.7 126.0,32.4 135.0,41.4 144.0,40.2 153.0,39.7 162.0,48.1 171.0,50.4 180.0,55.2 189.0,53.2 198.0,48.3 207.0,43.4 216.0,44.7 225.0,39.8 234.0,41.4 243.0,37.0 252.0,24.5 261.0,13.7 270.0,14.0" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-21.85s;--dur:11.8s"/>
<polyline points="0.0,91.5 9.0,79.0 18.0,81.2 27.0,71.5 36.0,69.9 45.0,58.7 54.0,46.6 63.0,44.1 72.0,47.0 81.0,55.7 90.0,49.7 99.0,42.7 108.0,30.5 117.0,20.9 126.0,22.5 135.0,12.1 144.0,21.8 153.0,13.4 162.0,7.0 171.0,17.0 180.0,22.5 189.0,16.1 198.0,18.8 207.0,16.2 216.0,16.8 225.0,11.5 234.0,7.0 243.0,8.1 252.0,7.0 261.0,7.0 270.0,7.0" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-23.00s;--dur:6.8s"/>
<polyline points="0.0,92.4 9.0,90.7 18.0,84.8 27.0,82.3 36.0,91.2 45.0,90.7 54.0,81.9 63.0,83.5 72.0,77.3 81.0,83.5 90.0,76.8 99.0,82.6 108.0,72.2 117.0,66.9 126.0,52.2 135.0,51.5 144.0,43.7 153.0,47.6 162.0,51.2 171.0,49.3 180.0,47.5 189.0,41.3 198.0,36.7 207.0,33.5 216.0,33.0 225.0,40.3 234.0,43.9 243.0,42.6 252.0,45.4 261.0,46.0 270.0,43.5" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-24.15s;--dur:8.1s"/>
</g></g>
<g transform="translate(1088 650)" opacity=".68"><g class="k-fin-viz">
<rect x="-10" y="-21" width="290" height="180" rx="12" fill="#0a1425" fill-opacity=".25" stroke="#456486" stroke-opacity=".2"/>
<text x="0" y="-6" fill="#a5c9e4" font-family="monospace" font-size="10" letter-spacing="1.4">MONTE CARLO / STOCHASTIC PATHS</text>
<path d="M0 150H270 M0 0V150" fill="none" stroke="#526781" stroke-opacity=".55" stroke-width=".8"/>
<polyline points="0.0,86.4 9.0,91.2 18.0,84.3 27.0,86.9 36.0,78.2 45.0,82.4 54.0,81.0 63.0,89.2 72.0,93.8 81.0,93.3 90.0,90.6 99.0,92.8 108.0,91.8 117.0,85.4 126.0,88.4 135.0,72.1 144.0,70.9 153.0,83.6 162.0,86.1 171.0,76.4 180.0,82.5 189.0,82.9 198.0,78.3 207.0,73.6 216.0,73.6 225.0,78.2 234.0,76.8 243.0,87.2 252.0,82.5 261.0,87.1 270.0,76.6" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-0.00s;--dur:6.8s"/>
<polyline points="0.0,79.0 9.0,77.6 18.0,78.6 27.0,89.7 36.0,86.9 45.0,88.0 54.0,79.6 63.0,79.1 72.0,81.9 81.0,78.7 90.0,84.5 99.0,92.8 108.0,95.9 117.0,94.9 126.0,93.0 135.0,96.3 144.0,89.2 153.0,94.7 162.0,90.3 171.0,91.4 180.0,83.8 189.0,77.3 198.0,71.5 207.0,74.2 216.0,73.9 225.0,73.7 234.0,79.4 243.0,73.1 252.0,78.1 261.0,73.7 270.0,67.4" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-1.15s;--dur:8.1s"/>
<polyline points="0.0,87.9 9.0,94.0 18.0,103.9 27.0,99.7 36.0,92.3 45.0,81.7 54.0,81.2 63.0,83.3 72.0,85.2 81.0,86.1 90.0,94.2 99.0,93.9 108.0,95.8 117.0,93.4 126.0,86.7 135.0,71.8 144.0,71.3 153.0,66.8 162.0,62.7 171.0,58.1 180.0,57.7 189.0,57.7 198.0,42.7 207.0,39.6 216.0,41.6 225.0,42.9 234.0,40.7 243.0,41.0 252.0,30.1 261.0,31.5 270.0,28.9" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-2.30s;--dur:9.3s"/>
<polyline points="0.0,91.6 9.0,94.9 18.0,94.9 27.0,86.8 36.0,73.3 45.0,69.5 54.0,68.6 63.0,69.1 72.0,74.7 81.0,76.4 90.0,77.5 99.0,65.4 108.0,69.5 117.0,69.7 126.0,82.1 135.0,86.1 144.0,80.5 153.0,80.2 162.0,83.4 171.0,76.7 180.0,65.4 189.0,67.4 198.0,63.2 207.0,61.1 216.0,60.8 225.0,58.2 234.0,62.4 243.0,63.3 252.0,64.3 261.0,68.9 270.0,67.8" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-3.45s;--dur:10.5s"/>
<polyline points="0.0,83.2 9.0,82.4 18.0,73.2 27.0,73.8 36.0,65.5 45.0,64.6 54.0,71.7 63.0,75.2 72.0,69.3 81.0,59.5 90.0,55.9 99.0,61.6 108.0,64.3 117.0,53.8 126.0,58.4 135.0,59.4 144.0,63.3 153.0,71.4 162.0,65.3 171.0,68.1 180.0,68.5 189.0,74.6 198.0,71.9 207.0,62.2 216.0,62.5 225.0,68.0 234.0,65.6 243.0,63.2 252.0,63.5 261.0,61.9 270.0,58.4" fill="none" stroke="#65cdb7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-4.60s;--dur:11.8s"/>
<polyline points="0.0,88.6 9.0,97.2 18.0,93.7 27.0,94.1 36.0,100.7 45.0,107.0 54.0,95.8 63.0,90.2 72.0,81.5 81.0,80.0 90.0,64.9 99.0,55.3 108.0,53.6 117.0,55.1 126.0,46.9 135.0,47.7 144.0,55.2 153.0,48.4 162.0,55.8 171.0,54.7 180.0,40.0 189.0,40.2 198.0,40.2 207.0,48.7 216.0,43.5 225.0,54.4 234.0,53.4 243.0,56.1 252.0,54.2 261.0,56.8 270.0,64.0" fill="none" stroke="#c4a7ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-5.75s;--dur:6.8s"/>
<polyline points="0.0,91.3 9.0,87.3 18.0,90.8 27.0,88.5 36.0,89.6 45.0,95.3 54.0,95.6 63.0,97.3 72.0,97.9 81.0,97.8 90.0,102.6 99.0,109.6 108.0,110.8 117.0,99.1 126.0,89.0 135.0,93.4 144.0,102.2 153.0,109.8 162.0,106.2 171.0,107.1 180.0,103.1 189.0,105.1 198.0,104.0 207.0,96.9 216.0,100.2 225.0,99.8 234.0,112.8 243.0,119.5 252.0,109.2 261.0,114.3 270.0,107.9" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-6.90s;--dur:8.1s"/>
<polyline points="0.0,87.5 9.0,85.7 18.0,76.8 27.0,95.7 36.0,100.4 45.0,94.1 54.0,100.8 63.0,85.8 72.0,93.1 81.0,93.3 90.0,94.4 99.0,99.8 108.0,100.6 117.0,103.8 126.0,97.3 135.0,99.6 144.0,97.3 153.0,100.2 162.0,98.6 171.0,100.9 180.0,97.2 189.0,88.7 198.0,83.5 207.0,89.1 216.0,87.1 225.0,87.4 234.0,84.8 243.0,90.2 252.0,84.7 261.0,67.5 270.0,62.7" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-8.05s;--dur:9.3s"/>
<polyline points="0.0,87.7 9.0,95.0 18.0,89.5 27.0,85.8 36.0,89.1 45.0,93.6 54.0,91.0 63.0,98.7 72.0,95.6 81.0,96.2 90.0,90.0 99.0,75.7 108.0,70.8 117.0,66.4 126.0,60.2 135.0,74.6 144.0,64.1 153.0,66.8 162.0,76.4 171.0,80.8 180.0,83.9 189.0,80.0 198.0,75.9 207.0,83.7 216.0,93.6 225.0,89.5 234.0,91.7 243.0,91.3 252.0,95.4 261.0,90.4 270.0,96.3" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-9.20s;--dur:10.5s"/>
<polyline points="0.0,89.1 9.0,90.5 18.0,85.8 27.0,92.1 36.0,89.6 45.0,93.0 54.0,92.3 63.0,97.2 72.0,99.7 81.0,115.2 90.0,109.9 99.0,114.9 108.0,111.2 117.0,111.6 126.0,100.3 135.0,107.7 144.0,92.2 153.0,96.7 162.0,86.4 171.0,84.0 180.0,86.7 189.0,81.2 198.0,83.0 207.0,82.2 216.0,77.6 225.0,79.8 234.0,88.9 243.0,90.6 252.0,90.4 261.0,90.8 270.0,90.0" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-10.35s;--dur:11.8s"/>
<polyline points="0.0,99.7 9.0,104.9 18.0,103.7 27.0,105.9 36.0,99.0 45.0,98.3 54.0,99.8 63.0,99.0 72.0,98.0 81.0,103.8 90.0,107.9 99.0,111.5 108.0,96.1 117.0,96.7 126.0,91.6 135.0,87.3 144.0,87.2 153.0,90.3 162.0,77.4 171.0,86.9 180.0,89.6 189.0,88.0 198.0,80.5 207.0,71.1 216.0,79.9 225.0,74.2 234.0,71.4 243.0,68.5 252.0,66.3 261.0,80.6 270.0,91.0" fill="none" stroke="#65cdb7" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-11.50s;--dur:6.8s"/>
<polyline points="0.0,84.1 9.0,77.4 18.0,86.1 27.0,86.8 36.0,79.7 45.0,75.2 54.0,80.5 63.0,78.4 72.0,77.2 81.0,100.4 90.0,102.8 99.0,96.4 108.0,99.2 117.0,91.7 126.0,86.7 135.0,83.3 144.0,75.0 153.0,76.0 162.0,75.4 171.0,74.1 180.0,79.2 189.0,85.9 198.0,84.4 207.0,91.8 216.0,99.8 225.0,96.3 234.0,101.6 243.0,93.9 252.0,85.5 261.0,84.9 270.0,90.7" fill="none" stroke="#c4a7ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-12.65s;--dur:8.1s"/>
<polyline points="0.0,106.1 9.0,106.2 18.0,113.7 27.0,104.6 36.0,106.2 45.0,114.0 54.0,109.8 63.0,110.5 72.0,112.6 81.0,124.4 90.0,116.8 99.0,110.6 108.0,115.3 117.0,120.9 126.0,123.4 135.0,119.9 144.0,120.5 153.0,134.5 162.0,132.6 171.0,138.7 180.0,144.0 189.0,140.9 198.0,137.9 207.0,142.4 216.0,141.5 225.0,142.5 234.0,138.1 243.0,136.7 252.0,135.3 261.0,144.0 270.0,144.0" fill="none" stroke="#6de4d7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-13.80s;--dur:9.3s"/>
<polyline points="0.0,93.0 9.0,100.7 18.0,102.7 27.0,102.0 36.0,106.2 45.0,95.1 54.0,92.9 63.0,94.2 72.0,93.0 81.0,91.5 90.0,91.5 99.0,85.0 108.0,91.7 117.0,87.3 126.0,92.2 135.0,85.2 144.0,101.2 153.0,105.3 162.0,91.9 171.0,80.8 180.0,71.1 189.0,68.5 198.0,65.4 207.0,75.2 216.0,85.9 225.0,75.2 234.0,70.9 243.0,57.4 252.0,51.6 261.0,52.0 270.0,54.7" fill="none" stroke="#a6a0ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-14.95s;--dur:10.5s"/>
<polyline points="0.0,93.2 9.0,102.6 18.0,95.3 27.0,89.9 36.0,91.0 45.0,96.6 54.0,95.0 63.0,97.0 72.0,107.1 81.0,103.0 90.0,103.8 99.0,97.0 108.0,102.3 117.0,91.5 126.0,92.3 135.0,107.3 144.0,107.2 153.0,104.4 162.0,105.7 171.0,106.5 180.0,103.8 189.0,99.7 198.0,99.9 207.0,100.3 216.0,94.4 225.0,95.7 234.0,91.1 243.0,88.7 252.0,85.1 261.0,75.7 270.0,61.7" fill="none" stroke="#70b9f3" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-16.10s;--dur:11.8s"/>
<polyline points="0.0,101.3 9.0,93.7 18.0,88.5 27.0,88.1 36.0,92.6 45.0,90.6 54.0,83.9 63.0,88.3 72.0,91.6 81.0,87.2 90.0,86.6 99.0,86.3 108.0,100.0 117.0,101.7 126.0,94.5 135.0,94.2 144.0,84.1 153.0,86.0 162.0,92.7 171.0,99.9 180.0,102.5 189.0,98.7 198.0,97.2 207.0,91.6 216.0,90.3 225.0,79.1 234.0,62.2 243.0,62.1 252.0,59.1 261.0,51.5 270.0,46.1" fill="none" stroke="#b0dbf4" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-17.25s;--dur:6.8s"/>
<polyline points="0.0,95.7 9.0,92.2 18.0,96.5 27.0,99.1 36.0,91.7 45.0,101.2 54.0,106.9 63.0,100.3 72.0,99.9 81.0,92.2 90.0,90.4 99.0,96.3 108.0,93.1 117.0,87.1 126.0,89.6 135.0,82.5 144.0,82.1 153.0,88.8 162.0,92.0 171.0,86.7 180.0,82.8 189.0,88.9 198.0,99.0 207.0,94.6 216.0,92.9 225.0,87.3 234.0,79.1 243.0,72.1 252.0,68.8 261.0,75.4 270.0,82.3" fill="none" stroke="#65cdb7" stroke-width="1.35" opacity="0.55" class="k-fin-path" style="--lag:-18.40s;--dur:8.1s"/>
<polyline points="0.0,103.1 9.0,97.7 18.0,88.8 27.0,92.4 36.0,81.7 45.0,75.9 54.0,82.4 63.0,88.4 72.0,94.5 81.0,88.2 90.0,92.9 99.0,94.7 108.0,89.1 117.0,92.6 126.0,102.4 135.0,98.1 144.0,110.6 153.0,113.1 162.0,110.7 171.0,95.7 180.0,89.1 189.0,82.9 198.0,74.7 207.0,78.2 216.0,82.6 225.0,81.9 234.0,80.7 243.0,78.6 252.0,72.4 261.0,68.9 270.0,79.9" fill="none" stroke="#c4a7ff" stroke-width="1.35" opacity="0.25" class="k-fin-path" style="--lag:-19.55s;--dur:9.3s"/>
</g></g>
<g transform="translate(1070 66)" opacity=".68"><g class="k-fin-viz">
<rect x="-10" y="-21" width="310" height="204" rx="12" fill="#0a1425" fill-opacity=".25" stroke="#456486" stroke-opacity=".2"/>
<text x="0" y="-6" fill="#a5c9e4" font-family="monospace" font-size="10" letter-spacing="1.4">EFFICIENT FRONTIER / CML</text>
<path d="M0 174H290 M0 0V174" fill="none" stroke="#526781" stroke-opacity=".55" stroke-width=".8"/>
<polyline points="0.0,112.0 3.9,107.9 7.7,103.9 11.6,100.0 15.5,96.1 19.3,92.3 23.2,88.6 27.1,84.9 30.9,81.3 34.8,77.8 38.7,74.3 42.5,70.9 46.4,67.6 50.3,64.3 54.1,61.1 58.0,57.9 61.9,54.8 65.7,51.8 69.6,48.9 73.5,46.0 77.3,43.2 81.2,40.4 85.1,37.7 88.9,35.1 92.8,32.6 96.7,30.1 100.5,27.7 104.4,25.3 108.3,23.0 112.1,20.8 116.0,18.6 119.9,16.6 123.7,14.5 127.6,12.6 131.5,10.7 135.3,8.8 139.2,7.1 143.1,5.4 146.9,3.7 150.8,2.2 154.7,0.7 158.5,-0.8 162.4,-2.1 166.3,-3.4 170.1,-4.7 174.0,-5.8 177.9,-6.9 181.7,-8.0 185.6,-9.0 189.5,-9.9 193.3,-10.7 197.2,-11.5 201.1,-12.2 204.9,-12.8 208.8,-13.4 212.7,-13.9 216.5,-14.4 220.4,-14.8 224.3,-15.1 228.1,-15.3 232.0,-15.5 235.9,-15.6 239.7,-15.7 243.6,-15.7 247.5,-15.6 251.3,-15.4 255.2,-15.2 259.1,-15.0 262.9,-14.6 266.8,-14.2 270.7,-13.7 274.5,-13.2 278.4,-12.6 282.3,-11.9 286.1,-11.2 290.0,-10.4" fill="none" stroke="#6ce1d4" stroke-width="1.35" opacity="0.94" class="k-fin-path" style="--lag:-0.00s;--dur:6.8s"/>
<line x1="0" y1="65.7" x2="290" y2="-56.6" stroke="#afa5ff" stroke-width="1.8" stroke-dasharray="5 5" class="k-cml"/>
<circle cx="145" cy="4.6" r="4.5" fill="#e3f9f8" class="k-tangent"/>
<text x="153" y="-5.4" fill="#b4f4e7" font-family="monospace" font-size="9">MAX SHARPE</text>
<text x="252" y="166" fill="#7a92b0" font-family="monospace" font-size="9">σ →</text>
</g></g>
<g transform="translate(60 620)" opacity=".68"><g class="k-fin-viz">
<rect x="-10" y="-21" width="300" height="185" rx="12" fill="#0a1425" fill-opacity=".25" stroke="#456486" stroke-opacity=".2"/>
<text x="0" y="-6" fill="#a5c9e4" font-family="monospace" font-size="10" letter-spacing="1.4">GAUSSIAN DISTRIBUTION / N(μ,σ²)</text>
<path d="M0 155H280 M0 0V155" fill="none" stroke="#526781" stroke-opacity=".55" stroke-width=".8"/>
<polygon points="0.0,144.0 0.0,143.3 2.5,143.1 5.1,143.0 7.6,142.8 10.2,142.5 12.7,142.3 15.3,141.9 17.8,141.6 20.4,141.2 22.9,140.7 25.5,140.2 28.0,139.5 30.5,138.8 33.1,138.0 35.6,137.2 38.2,136.2 40.7,135.1 43.3,133.8 45.8,132.5 48.4,131.0 50.9,129.3 53.5,127.5 56.0,125.6 58.5,123.5 61.1,121.2 63.6,118.7 66.2,116.1 67.4,144.0" fill="#aa84ff" opacity=".17"/>
<polyline points="0.0,143.3 2.5,143.1 5.1,143.0 7.6,142.8 10.2,142.5 12.7,142.3 15.3,141.9 17.8,141.6 20.4,141.2 22.9,140.7 25.5,140.2 28.0,139.5 30.5,138.8 33.1,138.0 35.6,137.2 38.2,136.2 40.7,135.1 43.3,133.8 45.8,132.5 48.4,131.0 50.9,129.3 53.5,127.5 56.0,125.6 58.5,123.5 61.1,121.2 63.6,118.7 66.2,116.1 68.7,113.3 71.3,110.3 73.8,107.2 76.4,103.9 78.9,100.5 81.5,97.0 84.0,93.3 86.5,89.5 89.1,85.6 91.6,81.7 94.2,77.7 96.7,73.7 99.3,69.7 101.8,65.8 104.4,61.9 106.9,58.1 109.5,54.4 112.0,50.9 114.5,47.6 117.1,44.5 119.6,41.6 122.2,39.0 124.7,36.7 127.3,34.7 129.8,33.0 132.4,31.7 134.9,30.8 137.5,30.2 140.0,30.0 142.5,30.2 145.1,30.8 147.6,31.7 150.2,33.0 152.7,34.7 155.3,36.7 157.8,39.0 160.4,41.6 162.9,44.5 165.5,47.6 168.0,50.9 170.5,54.4 173.1,58.1 175.6,61.9 178.2,65.8 180.7,69.7 183.3,73.7 185.8,77.7 188.4,81.7 190.9,85.6 193.5,89.5 196.0,93.3 198.5,97.0 201.1,100.5 203.6,103.9 206.2,107.2 208.7,110.3 211.3,113.3 213.8,116.1 216.4,118.7 218.9,121.2 221.5,123.5 224.0,125.6 226.5,127.5 229.1,129.3 231.6,131.0 234.2,132.5 236.7,133.8 239.3,135.1 241.8,136.2 244.4,137.2 246.9,138.0 249.5,138.8 252.0,139.5 254.5,140.2 257.1,140.7 259.6,141.2 262.2,141.6 264.7,141.9 267.3,142.3 269.8,142.5 272.4,142.8 274.9,143.0 277.5,143.1 280.0,143.3" fill="none" stroke="#8fe7df" stroke-width="1.35" opacity="0.98" class="k-fin-path" style="--lag:-2.30s;--dur:9.3s"/>
<line x1="140.0" y1="28" x2="140.0" y2="144" stroke="#d7b6ff" stroke-width="1" stroke-dasharray="4 5"/>
<text x="147.0" y="151" fill="#b6b1f3" font-family="monospace" font-size="10">μ</text>
<text x="5" y="151" fill="#bba5ee" font-family="monospace" font-size="9">VaR 5%</text>
</g></g>
<g transform="translate(655 708)" opacity=".68"><g class="k-fin-viz">
<rect x="-10" y="-21" width="242" height="175" rx="12" fill="#0a1425" fill-opacity=".25" stroke="#456486" stroke-opacity=".2"/>
<text x="0" y="-6" fill="#a5c9e4" font-family="monospace" font-size="10" letter-spacing="1.4">CORRELATION / Σ</text>
<path d="M0 145H222 M0 0V145" fill="none" stroke="#526781" stroke-opacity=".55" stroke-width=".8"/>
<rect x="10" y="10" width="23" height="14" rx="2" fill="#69dcd3" opacity="0.7" class="k-matrix-cell" style="--lag:-0.0s"/>
<rect x="39" y="10" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-0.2s"/>
<rect x="68" y="10" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-0.3s"/>
<rect x="97" y="10" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-0.5s"/>
<rect x="126" y="10" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-0.6s"/>
<rect x="155" y="10" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-0.8s"/>
<rect x="10" y="29" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-1.0s"/>
<rect x="39" y="29" width="23" height="14" rx="2" fill="#69dcd3" opacity="0.7" class="k-matrix-cell" style="--lag:-1.1s"/>
<rect x="68" y="29" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-1.3s"/>
<rect x="97" y="29" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-1.4s"/>
<rect x="126" y="29" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-1.6s"/>
<rect x="155" y="29" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-1.8s"/>
<rect x="10" y="48" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-1.9s"/>
<rect x="39" y="48" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-2.1s"/>
<rect x="68" y="48" width="23" height="14" rx="2" fill="#69dcd3" opacity="0.7" class="k-matrix-cell" style="--lag:-2.2s"/>
<rect x="97" y="48" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-2.4s"/>
<rect x="126" y="48" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-2.6s"/>
<rect x="155" y="48" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-2.7s"/>
<rect x="10" y="67" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-2.9s"/>
<rect x="39" y="67" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-3.0s"/>
<rect x="68" y="67" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-3.2s"/>
<rect x="97" y="67" width="23" height="14" rx="2" fill="#69dcd3" opacity="0.7" class="k-matrix-cell" style="--lag:-3.4s"/>
<rect x="126" y="67" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-3.5s"/>
<rect x="155" y="67" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-3.7s"/>
<rect x="10" y="86" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-3.8s"/>
<rect x="39" y="86" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-4.0s"/>
<rect x="68" y="86" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-4.2s"/>
<rect x="97" y="86" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-4.3s"/>
<rect x="126" y="86" width="23" height="14" rx="2" fill="#69dcd3" opacity="0.7" class="k-matrix-cell" style="--lag:-4.5s"/>
<rect x="155" y="86" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-4.6s"/>
<rect x="10" y="105" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-4.8s"/>
<rect x="39" y="105" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-5.0s"/>
<rect x="68" y="105" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-5.1s"/>
<rect x="97" y="105" width="23" height="14" rx="2" fill="#46749a" opacity="0.24" class="k-matrix-cell" style="--lag:-5.3s"/>
<rect x="126" y="105" width="23" height="14" rx="2" fill="#918bfa" opacity="0.42" class="k-matrix-cell" style="--lag:-5.4s"/>
<rect x="155" y="105" width="23" height="14" rx="2" fill="#69dcd3" opacity="0.7" class="k-matrix-cell" style="--lag:-5.6s"/>
</g></g>
<g transform="translate(415 70)" opacity=".68"><g class="k-fin-viz">
<rect x="-10" y="-21" width="238" height="135" rx="12" fill="#0a1425" fill-opacity=".25" stroke="#456486" stroke-opacity=".2"/>
<text x="0" y="-6" fill="#a5c9e4" font-family="monospace" font-size="10" letter-spacing="1.4">RETURN HISTOGRAM</text>
<path d="M0 105H218 M0 0V105" fill="none" stroke="#526781" stroke-opacity=".55" stroke-width=".8"/>
<rect class="k-vol-bar" style="--lag:-0.00s" x="3.0" y="99.5" width="7.5" height="5.5" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-0.19s" x="13.5" y="95.3" width="7.5" height="9.7" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-0.38s" x="24.0" y="87.0" width="7.5" height="18.0" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-0.57s" x="34.5" y="83.9" width="7.5" height="21.1" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-0.76s" x="45.0" y="67.9" width="7.5" height="37.1" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-0.95s" x="55.5" y="65.5" width="7.5" height="39.5" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-1.14s" x="66.0" y="50.2" width="7.5" height="54.8" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-1.33s" x="76.5" y="52.2" width="7.5" height="52.8" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-1.52s" x="87.0" y="22.5" width="7.5" height="82.5" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-1.71s" x="97.5" y="31.4" width="7.5" height="73.6" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-1.90s" x="108.0" y="30.6" width="7.5" height="74.4" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-2.09s" x="118.5" y="22.9" width="7.5" height="82.1" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-2.28s" x="129.0" y="44.0" width="7.5" height="61.0" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-2.47s" x="139.5" y="61.0" width="7.5" height="44.0" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-2.66s" x="150.0" y="65.2" width="7.5" height="39.8" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-2.85s" x="160.5" y="78.6" width="7.5" height="26.4" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-3.04s" x="171.0" y="86.1" width="7.5" height="18.9" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-3.23s" x="181.5" y="87.7" width="7.5" height="17.3" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-3.42s" x="192.0" y="94.3" width="7.5" height="10.7" rx="1" fill="#7896e9" opacity=".5"/>
<rect class="k-vol-bar" style="--lag:-3.61s" x="202.5" y="98.7" width="7.5" height="6.3" rx="1" fill="#7896e9" opacity=".5"/>
</g></g>
<text class="formula " style="--delay:-0.0s" x="42" y="278" font-size="14" opacity=".35">σ²ₚ = Σᵢ Σⱼ wᵢwⱼσᵢⱼ</text>
<text class="formula teal" style="--delay:-1.7s" x="365" y="105" font-size="13" opacity=".35">CML: E[Rₚ] = rᶠ + SRₘₐₓ · σₚ</text>
<text class="formula " style="--delay:-3.4s" x="1120" y="285" font-size="13" opacity=".35">w* = argmax (μₚ − rᶠ) / σₚ</text>
<text class="formula teal" style="--delay:-5.1s" x="85" y="510" font-size="14" opacity=".35">dSₜ = μSₜdt + σSₜdWₜ</text>
<text class="formula " style="--delay:-6.8s" x="1120" y="505" font-size="16" opacity=".35">VaRα = −(μ + zασ)</text>
<text class="formula teal" style="--delay:-8.5s" x="360" y="825" font-size="18" opacity=".35">P(R ≤ x) = Φ((x−μ)/σ)</text>
<text class="formula " style="--delay:-10.2s" x="825" y="515" font-size="14" opacity=".35">hₜ = ω + αε²ₜ₋₁ + βhₜ₋₁</text>
<text class="formula teal" style="--delay:-11.9s" x="1125" y="820" font-size="16" opacity=".35">ESα = E[L | L ≥ VaRα]</text>
<text class="formula " style="--delay:-13.6s" x="160" y="610" font-size="13" opacity=".35">Rₚ = Σᵢ wᵢRᵢ</text>
<text class="formula teal" style="--delay:-15.3s" x="55" y="850" font-size="14" opacity=".35">Sharpe = (μ − rᶠ)/σ</text>
<text class="formula " style="--delay:-17.0s" x="900" y="850" font-size="16" opacity=".35">ρᵢⱼ = σᵢⱼ/(σᵢσⱼ)</text>
<text class="formula teal" style="--delay:-18.7s" x="750" y="65" font-size="13" opacity=".35">E[Rᵢ] = rᶠ + βᵢ(E[Rₘ]−rᶠ)</text>
      </svg>
    </div>
    """).strip(), unsafe_allow_html=True)

    st.markdown('<div class="k-brand">◈ &nbsp; KANGEMI EDU <span>/ QUANTITATIVE BOUTIQUE</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, .82], gap="large", vertical_alignment="center")
    with left:
        st.markdown(re.sub(r">\s+<", "><", """
        <div class="k-eyebrow">Research · Portfolio Engineering · Risk Intelligence</div>
        <div class="k-title">Invest with<br><em>quantitative<br>clarity.</em></div>
        <div class="k-sub">Dalla ricerca quantitativa alla costruzione del portafoglio: un ambiente integrato per analizzare i mercati, sviluppare strategie di investimento e valutare opportunità, performance e rischi attraverso modelli finanziari avanzati.</div>
        <div class="k-chips">
          <span class="k-chip">MARKOWITZ</span> <span class="k-chip">CAPM</span> <span class="k-chip">FAMA–FRENCH</span>
          <span class="k-chip">GARCH</span><span class="k-chip">MONTE CARLO</span> <span class="k-chip">BACKTESTING</span> <span class="k-chip">ROBUSTNESS CHECK</span>
        </div>
        """).strip(), unsafe_allow_html=True)
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
