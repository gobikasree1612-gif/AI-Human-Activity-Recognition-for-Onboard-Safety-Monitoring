"""
dashboard/ui.py
===============
Reusable Streamlit UI component renderers.

All rendering functions are pure: they take data, render it, and return nothing.
They do NOT read from session_state directly (passed as arguments).
This makes them easy to test and reason about.
"""

import base64
import datetime
import io
import math
import struct
from typing import Dict, List, Optional
import wave

import streamlit as st


def generate_warning_beep_wav_b64() -> str:
    """Generate a short non-looping warning beep WAV audio encoded as base64 data URI."""
    sample_rate = 22050
    duration = 0.35  # seconds
    n_samples = int(sample_rate * duration)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            volume_env = math.sin(math.pi * t / duration) ** 0.5
            val = math.sin(2 * math.pi * 880 * t) * 0.7 * volume_env
            sample_16bit = int(val * 32767)
            samples.append(struct.pack("<h", sample_16bit))

        wav_file.writeframes(b"".join(samples))

    wav_bytes = buf.getvalue()
    b64_str = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64_str}"



# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

DARK_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════
   ONBOARD HAR — AEROSPACE DARK THEME
   ═══════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0E1A !important;
    color: #C9D1D9 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #0A0E1A 100%) !important;
    border-right: 1px solid #1E2D3D !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0D1626 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 6px !important;
    padding: 12px !important;
}

[data-testid="metric-container"] label {
    color: #8B9DC3 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00C896 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #1E2D3D !important;
    color: #8B9DC3 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #00C896 !important;
    color: #00C896 !important;
    box-shadow: 0 0 12px rgba(0,200,150,0.15) !important;
}

/* Sidebar text */
.sidebar-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: #00C896;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid #1E2D3D;
    margin-bottom: 6px;
}
.sidebar-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #4A6080;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Section labels */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.70rem;
    color: #4A6080;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid #1A2332;
    padding-bottom: 4px;
}

/* Person cards */
.person-card {
    background: #0D1626;
    border: 1px solid #1E2D3D;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
}
.person-card.alert-warning {
    border-color: #D4A017 !important;
    box-shadow: 0 0 14px rgba(212,160,23,0.18);
}
.person-card.alert-critical {
    border-color: #CC2936 !important;
    box-shadow: 0 0 16px rgba(204,41,54,0.22);
}
.person-id {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #8B9DC3;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.activity-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 2px 10px;
    border-radius: 3px;
    margin-left: 8px;
}
.evidence-item {
    font-size: 0.78rem;
    color: #8B9DC3;
    padding: 2px 0;
}

/* Status banners */
.status-normal {
    background: #051A10;
    border: 1px solid #005533;
    border-radius: 6px;
    padding: 14px 20px;
    text-align: center;
}
.status-warning {
    background: #1A1200;
    border: 1px solid #665500;
    border-radius: 6px;
    padding: 14px 20px;
    text-align: center;
    animation: pulse-warning 2s infinite;
}
.status-critical {
    background: #1A0005;
    border: 1px solid #8B0000;
    border-radius: 6px;
    padding: 14px 20px;
    text-align: center;
    animation: pulse-critical 1s infinite;
}
@keyframes pulse-warning {
    0%, 100% { box-shadow: 0 0 10px rgba(212,160,23,0.2); }
    50%       { box-shadow: 0 0 22px rgba(212,160,23,0.45); }
}
@keyframes pulse-critical {
    0%, 100% { box-shadow: 0 0 12px rgba(204,41,54,0.3); }
    50%       { box-shadow: 0 0 28px rgba(204,41,54,0.65); }
}
.status-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #4A6080;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.status-value-normal   { font-family: 'Share Tech Mono', monospace; font-size: 1.4rem; color: #00C896; font-weight: 700; letter-spacing: 0.1em; }
.status-value-warning  { font-family: 'Share Tech Mono', monospace; font-size: 1.4rem; color: #F5A623; font-weight: 700; letter-spacing: 0.1em; }
.status-value-critical { font-family: 'Share Tech Mono', monospace; font-size: 1.4rem; color: #FF3B3B; font-weight: 700; letter-spacing: 0.1em; }

/* Alert box */
.fall-alert-critical {
    background: linear-gradient(135deg, #1A0008, #200010);
    border: 2px solid #CC2936;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 10px 0;
    animation: pulse-critical 1s infinite;
}
.fall-alert-warning {
    background: linear-gradient(135deg, #1A1000, #1E1400);
    border: 2px solid #D4A017;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    animation: pulse-warning 2s infinite;
}

/* Online indicator */
.online-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00C896;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 8px rgba(0,200,150,0.6);
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* Header */
.main-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    color: #C9D1D9;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0;
}
.main-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #4A6080;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 2px 0 0 0;
}

/* DataFrame table */
[data-testid="stDataFrame"] {
    border: 1px solid #1E2D3D !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #1E2D3D !important;
    border-radius: 6px !important;
    background: #0D1117 !important;
}

/* Selectbox / sliders */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stFileUploader"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #8B9DC3 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Divider */
hr { border-color: #1E2D3D !important; }

/* Hide Streamlit default extras */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""


# ---------------------------------------------------------------------------
# Component: page CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Component: top header bar
# ---------------------------------------------------------------------------

def render_header() -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown(
            '<p class="main-title">Onboard HAR Safety Monitor</p>'
            '<p class="main-subtitle">Real-Time AI-Based Human Activity &amp; Fall Safety Monitoring System</p>',
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            f'<div style="text-align:right; padding-top:10px;">'
            f'<span class="online-dot"></span>'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;'
            f'color:#00C896;letter-spacing:0.1em;">SYSTEM ONLINE</span><br>'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
            f'color:#4A6080;">{now}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Component: top system status cards
# ---------------------------------------------------------------------------

def render_top_system_cards(
    monitoring: bool,
    camera_connected: bool,
    classifier_mode: str,
    system_status: str,
) -> None:
    c1, c2, c3, c4 = st.columns(4)

    # Card 1: Camera Status
    if monitoring and camera_connected:
        cam_html = '<span style="color:#00C896;font-weight:700;">● Connected</span>'
    elif monitoring and not camera_connected:
        cam_html = '<span style="color:#FF3B3B;font-weight:700;">⚠ Error</span>'
    else:
        cam_html = '<span style="color:#6B7280;">○ Disconnected</span>'

    with c1:
        st.markdown(
            f'<div style="background:#0D1626;border:1px solid #1E2D3D;border-radius:6px;padding:10px 14px;">'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#4A6080;letter-spacing:0.12em;">CAMERA SOURCE</div>'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.05rem;margin-top:2px;">{cam_html}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 2: Pose Detection
    pose_html = '<span style="color:#00C896;font-weight:700;">● Active (YOLO11)</span>' if monitoring else '<span style="color:#6B7280;">○ Standby</span>'
    with c2:
        st.markdown(
            f'<div style="background:#0D1626;border:1px solid #1E2D3D;border-radius:6px;padding:10px 14px;">'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#4A6080;letter-spacing:0.12em;">POSE DETECTOR</div>'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.05rem;margin-top:2px;">{pose_html}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 3: HAR Model
    if classifier_mode == "trained_ml":
        model_html = '<span style="color:#00C896;font-weight:700;">Trained ML Model</span>'
    else:
        model_html = '<span style="color:#F5A623;font-weight:700;">Rule-based / heuristic fallback</span>'

    with c3:
        st.markdown(
            f'<div style="background:#0D1626;border:1px solid #1E2D3D;border-radius:6px;padding:10px 14px;">'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#4A6080;letter-spacing:0.12em;">HAR MODEL</div>'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.82rem;margin-top:4px;">{model_html}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 4: Safety Status
    if system_status == "CRITICAL":
        status_html = '<span style="color:#FF3B3B;font-weight:700;">🚨 FALL DETECTED</span>'
        border_col = "#8B0000"
    elif system_status == "WARNING":
        status_html = '<span style="color:#F5A623;font-weight:700;">⚠ WARNING</span>'
        border_col = "#665500"
    else:
        status_html = '<span style="color:#00C896;font-weight:700;">● NORMAL</span>'
        border_col = "#005533"

    with c4:
        st.markdown(
            f'<div style="background:#0D1626;border:1px solid {border_col};border-radius:6px;padding:10px 14px;">'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#4A6080;letter-spacing:0.12em;">SAFETY STATUS</div>'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.05rem;margin-top:2px;">{status_html}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Component: system status banner
# ---------------------------------------------------------------------------

def render_system_status(system_status: str) -> None:
    if system_status == "CRITICAL":
        label_cls = "status-value-critical"
        box_cls   = "status-critical"
        status_text = "⚠ FALL DETECTED"
    elif system_status == "WARNING":
        label_cls = "status-value-warning"
        box_cls   = "status-warning"
        status_text = "⚠ POSSIBLE FALL DETECTED"
    else:
        label_cls = "status-value-normal"
        box_cls   = "status-normal"
        status_text = "● NORMAL"

    st.markdown(
        f'<div class="{box_cls}">'
        f'<div class="status-label">SYSTEM SAFETY STATUS</div>'
        f'<div class="{label_cls}">{status_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component: live metric cards
# ---------------------------------------------------------------------------

def render_metric_cards(
    people_count: int,
    fps: float,
    latency_ms: float,
    alert_count: int,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 PEOPLE DETECTED", people_count)
    c2.metric("⚡ INFERENCE FPS", f"{fps:.1f}")
    c3.metric("⏱ LATENCY", f"{latency_ms:.0f} ms")
    c4.metric("🚨 FALL ALERTS", alert_count)


# ---------------------------------------------------------------------------
# Component: per-person activity card
# ---------------------------------------------------------------------------

_ACTIVITY_COLOR = {
    "STANDING": "#00C896",
    "SITTING":  "#4A9EFF",
    "WALKING":  "#F5A623",
    "RUNNING":  "#E05CF5",
    "FALLING":  "#FF3B3B",
    "UNKNOWN":  "#6B7280",
}

_MODEL_LABEL = {
    "trained_ml":          "Trained ML Model",
    "rule_based_baseline": "Rule-based / heuristic fallback",
}


def render_person_card(person: dict) -> None:
    pid      = person.get("person_id", "?")
    activity = person.get("activity", "UNKNOWN")
    conf     = person.get("confidence", 0.0)
    severity = person.get("severity")
    alert    = person.get("alert", False)
    alert_type = person.get("alert_type")
    model_type = person.get("model_type", "rule_based_baseline")
    evidence = person.get("evidence", [])

    card_cls = (
        "person-card alert-critical" if severity == "CRITICAL"
        else "person-card alert-warning" if severity == "WARNING"
        else "person-card"
    )
    activity_color = _ACTIVITY_COLOR.get(activity, "#6B7280")
    model_label    = _MODEL_LABEL.get(model_type, model_type)

    # Status indicator
    if severity == "CRITICAL":
        status_html = '<span style="color:#FF3B3B;font-size:0.75rem;font-weight:700;">⚠ CRITICAL</span>'
    elif severity == "WARNING":
        status_html = '<span style="color:#F5A623;font-size:0.75rem;font-weight:700;">⚠ WARNING</span>'
    else:
        status_html = '<span style="color:#00C896;font-size:0.75rem;">● NORMAL</span>'

    # Evidence list
    ev_html = "".join(
        f'<div class="evidence-item">✓ {e}</div>' for e in evidence
    ) if evidence else '<div class="evidence-item" style="color:#4A6080;">—</div>'

    # Alert badge
    alert_badge = ""
    if alert_type == "FALL_DETECTED":
        alert_badge = '<span style="background:#CC2936;color:#fff;font-size:0.65rem;padding:2px 8px;border-radius:3px;margin-left:8px;font-family:monospace;">FALL DETECTED</span>'
    elif alert_type == "POSSIBLE_FALL":
        alert_badge = '<span style="background:#665500;color:#F5A623;font-size:0.65rem;padding:2px 8px;border-radius:3px;margin-left:8px;font-family:monospace;">POSSIBLE FALL</span>'

    st.markdown(
        f"""
<div class="{card_cls}">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <span class="person-id">PERSON #{pid}</span>
    {status_html}
  </div>
  <div style="display:flex;align-items:center;margin-bottom:6px;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:1.0rem;
                 color:{activity_color};font-weight:700;letter-spacing:0.05em;">
      {activity}
    </span>
    {alert_badge}
    <span style="margin-left:auto;font-family:'Share Tech Mono',monospace;
                 font-size:0.85rem;color:#8B9DC3;">
      {conf:.0%}
    </span>
  </div>
  <div style="font-size:0.68rem;color:#4A6080;font-family:monospace;
              margin-bottom:8px;letter-spacing:0.08em;">
    MODEL: {model_label.upper()}
  </div>
  <div style="font-size:0.7rem;color:#4A6080;font-family:monospace;
              letter-spacing:0.08em;margin-bottom:4px;">EVIDENCE</div>
  {ev_html}
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component: fall alert panel
# ---------------------------------------------------------------------------

def render_fall_alert(
    people: List[dict],
    latched_fall: Optional[dict] = None,
) -> None:
    critical_people = [p for p in people if p.get("severity") == "CRITICAL"]
    warning_people  = [p for p in people if p.get("severity") == "WARNING"]

    if not critical_people and not warning_people and not latched_fall:
        return

    if latched_fall:
        evidence_html = "".join(
            f'<li style="color:#FF9999;font-size:0.78rem;margin:2px 0;">✓ {e}</li>'
            for e in latched_fall.get("evidence", [])
        )
        pid = latched_fall.get("person_id", "?")
        ts = latched_fall.get("timestamp", datetime.datetime.now().strftime("%H:%M:%S"))
        st.markdown(
            f"""
<div class="fall-alert-critical">
  <div style="font-family:'Share Tech Mono',monospace;font-size:1.0rem;
              color:#FF3B3B;font-weight:700;letter-spacing:0.08em;margin-bottom:6px;">
    ⚠ FALL DETECTED — PERSON #{pid}
  </div>
  <div style="font-size:0.7rem;color:#CC2936;font-family:monospace;
              letter-spacing:0.1em;margin-bottom:10px;">
    SEVERITY: CRITICAL &nbsp;|&nbsp; DETECTED AT: {ts} &nbsp;|&nbsp; [LATCHED UNTIL RESET]
  </div>
  <ul style="margin:0;padding-left:18px;">
    {evidence_html}
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    for p in critical_people:
        evidence_html = "".join(
            f'<li style="color:#FF9999;font-size:0.78rem;margin:2px 0;">✓ {e}</li>'
            for e in p.get("evidence", [])
        )
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
<div class="fall-alert-critical">
  <div style="font-family:'Share Tech Mono',monospace;font-size:1.0rem;
              color:#FF3B3B;font-weight:700;letter-spacing:0.08em;margin-bottom:6px;">
    ⚠ FALL DETECTED — PERSON #{p.get('person_id','?')}
  </div>
  <div style="font-size:0.7rem;color:#CC2936;font-family:monospace;
              letter-spacing:0.1em;margin-bottom:10px;">
    SEVERITY: CRITICAL &nbsp;|&nbsp; {ts}
  </div>
  <ul style="margin:0;padding-left:18px;">
    {evidence_html}
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    for p in warning_people:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
<div class="fall-alert-warning">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.9rem;
              color:#F5A623;font-weight:700;letter-spacing:0.08em;margin-bottom:4px;">
    ⚠ POSSIBLE FALL — PERSON #{p.get('person_id','?')}
  </div>
  <div style="font-size:0.72rem;color:#997700;font-family:monospace;margin-bottom:2px;">
    Monitoring confirmation... &nbsp;|&nbsp; {ts}
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )



# ---------------------------------------------------------------------------
# Component: sidebar info block
# ---------------------------------------------------------------------------

def render_sidebar_system_info(
    classifier_mode: str,
    monitoring: bool,
) -> None:
    model_label = (
        "Trained ML" if classifier_mode == "trained_ml"
        else "Rule-Based Baseline"
    )
    model_color = "#00C896" if classifier_mode == "trained_ml" else "#F5A623"
    mon_label   = "● ACTIVE" if monitoring else "○ STOPPED"
    mon_color   = "#00C896" if monitoring else "#6B7280"

    st.markdown(
        f"""
<div style="background:#0D1117;border:1px solid #1E2D3D;border-radius:6px;
            padding:12px;margin-top:10px;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
              color:#4A6080;letter-spacing:0.14em;text-transform:uppercase;
              margin-bottom:8px;">SYSTEM INFORMATION</div>

  <div style="margin-bottom:6px;">
    <span style="font-size:0.65rem;color:#4A6080;font-family:monospace;">AI ENGINE</span><br>
    <span style="font-size:0.78rem;color:#C9D1D9;font-family:monospace;">YOLO Pose + Temporal HAR</span>
  </div>
  <div style="margin-bottom:6px;">
    <span style="font-size:0.65rem;color:#4A6080;font-family:monospace;">ACTIVITY CLASSES</span><br>
    <span style="font-size:0.78rem;color:#C9D1D9;font-family:monospace;">5</span>
  </div>
  <div style="margin-bottom:6px;">
    <span style="font-size:0.65rem;color:#4A6080;font-family:monospace;">MODE</span><br>
    <span style="font-size:0.78rem;color:#C9D1D9;font-family:monospace;">Edge / Local</span>
  </div>
  <div style="margin-bottom:6px;">
    <span style="font-size:0.65rem;color:#4A6080;font-family:monospace;">ACTIVITY MODEL</span><br>
    <span style="font-size:0.78rem;font-family:monospace;color:{model_color};">{model_label}</span>
  </div>
  <div>
    <span style="font-size:0.65rem;color:#4A6080;font-family:monospace;">MONITORING</span><br>
    <span style="font-size:0.78rem;font-family:monospace;color:{mon_color};">{mon_label}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component: section separator label
# ---------------------------------------------------------------------------

def section(label: str) -> None:
    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component: footer
# ---------------------------------------------------------------------------

def render_footer() -> None:
    st.markdown("<hr style='margin:20px 0 10px 0;'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;font-size:0.65rem;color:#2A3A50;'
        'font-family:monospace;letter-spacing:0.08em;">'
        '🔒 Processing runs locally. Video is not uploaded to any cloud service. '
        '&nbsp;|&nbsp; Onboard HAR © 2026 Hackathon Team'
        '</div>',
        unsafe_allow_html=True,
    )
