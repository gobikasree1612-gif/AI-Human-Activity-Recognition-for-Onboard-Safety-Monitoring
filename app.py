"""
app.py
======
ONBOARD HUMAN ACTIVITY & SAFETY MONITOR
Real-Time Streamlit Dashboard

Run:
    streamlit run app.py

Consumes the existing HAREngine:
    from ai.inference import HAREngine
    result, annotated_frame = engine.process_frame(frame)

The dashboard does NOT re-implement any AI logic.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Dashboard modules ─────────────────────────────────────────────────────
from dashboard.analytics import (
    activity_distribution_chart,
    activity_timeline_chart,
    performance_chart,
)
from dashboard.camera import CameraSource, save_uploaded_video
from dashboard.history import SessionHistory
from dashboard.ui import (
    inject_css,
    render_fall_alert,
    render_footer,
    render_header,
    render_metric_cards,
    render_person_card,
    render_sidebar_system_info,
    render_system_status,
    render_top_system_cards,
    section,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Onboard HAR Monitor",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ═════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ═════════════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    """Initialise all session-state keys with safe defaults."""
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "history" not in st.session_state:
        st.session_state.history = SessionHistory()
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False
    if "camera" not in st.session_state:
        st.session_state.camera = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "engine_error" not in st.session_state:
        st.session_state.engine_error = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "classifier_mode" not in st.session_state:
        st.session_state.classifier_mode = "rule_based_baseline"
    if "latched_fall" not in st.session_state:
        st.session_state.latched_fall = None
    if "fall_sound_triggered" not in st.session_state:
        st.session_state.fall_sound_triggered = False


_init_session_state()



# ═════════════════════════════════════════════════════════════════════════
# Engine loader (cached per session)
# ═════════════════════════════════════════════════════════════════════════

def _load_engine():
    """Load HAREngine once per session. Stores in st.session_state.engine."""
    if st.session_state.engine is not None:
        return st.session_state.engine

    try:
        from ai.inference import HAREngine
        with st.spinner("Loading AI engine (YOLO Pose)..."):
            engine = HAREngine(config_path=os.path.join(_HERE, "config.yaml"))
        st.session_state.engine = engine
        st.session_state.classifier_mode = engine.classifier_mode
        st.session_state.engine_error = None
        logger.info("HAREngine loaded. Classifier: %s", engine.classifier_mode)
        return engine
    except Exception as exc:
        msg = f"Failed to load AI engine: {exc}"
        st.session_state.engine_error = msg
        logger.error(msg)
        return None


# ═════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════

def _render_sidebar(engine) -> dict:
    """Render sidebar and return settings dict."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">ONBOARD HAR</div>'
            '<div class="sidebar-sub">AI Human Activity Recognition</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Source ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">INPUT SOURCE</div>', unsafe_allow_html=True)
        source_type = st.selectbox(
            "Source type",
            ["Webcam", "Demo Dataset Video", "Upload Video File"],
            label_visibility="collapsed",
        )

        video_path = None
        cam_index = 0

        if source_type == "Webcam":
            cam_index = st.number_input(
                "Camera index", min_value=0, max_value=10, value=0, step=1
            )
        elif source_type == "Demo Dataset Video":
            raw_dir = os.path.join(_HERE, "dataset", "raw")
            activity_classes = ["STANDING", "SITTING", "WALKING", "RUNNING", "FALLING"]
            selected_class = st.selectbox("Select Activity Class", activity_classes)
            class_dir = os.path.join(raw_dir, selected_class)
            demo_videos = []
            if os.path.exists(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith((".mp4", ".avi", ".mov")):
                        rel = os.path.relpath(os.path.join(class_dir, f), _HERE).replace("\\", "/")
                        demo_videos.append(rel)
            demo_videos.sort()
            if demo_videos:
                selected_demo = st.selectbox("Select Dataset Clip", demo_videos)
                video_path = os.path.normpath(os.path.join(_HERE, selected_demo))
                st.session_state.video_path = video_path
            else:
                st.info(f"No recorded clips found for {selected_class}.")
        else:
            uploaded = st.file_uploader(
                "Upload video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
            )
            if uploaded is not None:
                if (
                    st.session_state.video_path is None
                    or uploaded.name not in str(st.session_state.video_path)
                ):
                    saved = save_uploaded_video(uploaded)
                    if saved:
                        st.session_state.video_path = saved
                video_path = st.session_state.video_path
            else:
                st.session_state.video_path = None

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Settings ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">SETTINGS</div>', unsafe_allow_html=True)
        show_pose  = st.toggle("Show pose keypoints", value=True)
        show_bbox  = st.toggle("Show bounding boxes", value=True)
        show_evidence = st.toggle("Show evidence panel", value=True)
        enable_alerts = st.toggle("Enable alerts", value=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Controls ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">CONTROLS</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            start_btn = st.button("▶  START", use_container_width=True)
        with col_b:
            stop_btn = st.button("■  STOP", use_container_width=True)

        reset_btn = st.button("↺  RESET SESSION", use_container_width=True)

        # ── Button logic ──────────────────────────────────────────────────
        if start_btn:
            source = cam_index if source_type == "Webcam" else video_path
            if source is None and source_type != "Webcam":
                st.warning("Please select or upload a valid video file first.")
            else:
                _start_monitoring(source, engine)

        if stop_btn and st.session_state.monitoring:
            _stop_monitoring()

        if reset_btn:
            _reset_session()

        # ── System info ───────────────────────────────────────────────────
        render_sidebar_system_info(
            classifier_mode=st.session_state.classifier_mode,
            monitoring=st.session_state.monitoring,
        )

        # ── Diagnostics (debug mode) ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">AI DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        with st.expander("View Pipeline Details", expanded=False):
            if engine is not None:
                try:
                    diag = engine.get_diagnostics()
                    det = diag.get("detector", {})
                    clf = diag.get("classifier", {})
                    fd  = diag.get("fall_detector", {})
                    pip = diag.get("pipeline", {})
                    perf = diag.get("performance", {})

                    from tools.dataset_assistant import inspect_readiness
                    readiness = inspect_readiness()
                    targets = readiness["targets"]
                    tgt_vids = targets["target_videos_per_class"]

                    # Build progress bar representation for each class
                    bars = []
                    for act in ["STANDING", "SITTING", "WALKING", "RUNNING", "FALLING"]:
                        st_info = readiness["class_status"].get(act, {})
                        cnt = st_info.get("videos", 0)
                        filled = min(5, int(round((cnt / tgt_vids) * 5))) if tgt_vids > 0 else 0
                        bar_str = "█" * filled + "░" * (5 - filled)
                        bars.append(f"{act:<9} {bar_str} ({cnt}/{tgt_vids})")
                    bar_display = "<br>".join(bars)

                    no_data_msg = """
<span style='color:#FFB800;'>No real recordings available.<br>
Use the dataset recording tool to begin collection.</span><br><br>
<span style='color:#FF6B6B;font-weight:600;'>TRAINING READINESS: BLOCKED</span><br>
<span style='color:#8B9DC3;'>Reason: No real recordings available.</span>
""" if readiness["total_videos"] == 0 else f"""
<span style='color:{"#00D4AA" if readiness["training_ready"] else "#FF6B6B"};font-weight:600;'>TRAINING READINESS: {"OPEN" if readiness["training_ready"] else "BLOCKED"}</span><br>
{f"<span style='color:#FFB800;'>Reason: {', '.join(readiness['next_actions'][:2])}</span><br>" if not readiness["training_ready"] else ""}
"""

                    st.markdown(
                        f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
            color:#8B9DC3;line-height:1.9;">
<span style="color:#4A6080;">── DATA COLLECTION ──</span><br>
Videos: {readiness['total_videos']}<br>
Subjects: {readiness['total_subjects']}<br>
Sessions: {readiness['total_sessions']}<br>
Total duration: {readiness['total_duration']:.0f} sec<br>
<br>
<span style="color:#4A6080;">── CLASS PROGRESS ──</span><br>
{bar_display}<br>
<br>
{no_data_msg}<br>

<span style="color:#4A6080;">── DETECTOR ──</span><br>
Model: {det.get('model_name','?')}<br>
Type: {det.get('model_type','?')}<br>
Det. conf: {det.get('detection_confidence_threshold','?')}<br>
IoU thresh: {det.get('iou_threshold','?')}<br>
Tracker: {det.get('tracker_type','?')}<br>
Keypoints: {det.get('pose_keypoints','?')} (COCO-17)<br>
Device: {det.get('inference_device','?')}<br>
Multi-person: {'✓' if det.get('multi_person_supported') else '✗'}<br>
<br>
<span style="color:#4A6080;">── CLASSIFIER ──</span><br>
Mode: <span style="color:{'#00D4AA' if clf.get('is_trained') else '#FFB800'};font-weight:600;">{'TRAINED ML' if clf.get('is_trained') else 'RULE-BASED FALLBACK'}</span><br>
Model version: {clf.get('model_version', 'N/A')}<br>
Feature dim: {clf.get('feature_dim', '?')}<br>
Seq length: {clf.get('sequence_length', '?')} frames<br>
Activity classes: {', '.join(clf.get('activities_supported', []))}<br>
{f"Load status: Rejected ({clf.get('load_error')})<br>" if clf.get('load_error') else ""}\
{f"Trained at: {clf.get('trained_at', 'N/A')}<br>Train samples: {clf.get('training_samples', 'N/A')}<br>Split: {clf.get('split_strategy', 'N/A')}<br>" if clf.get('is_trained') else ""}\
<br>
<span style="color:#4A6080;">── FALL DETECTOR ──</span><br>
Angle thresh: {fd.get('body_angle_threshold','?')}°<br>
Vel thresh: {fd.get('downward_velocity_threshold','?')}<br>
Confirm frames: {fd.get('confirmation_frames','?')}<br>
Timeout: {fd.get('possible_fall_timeout_s','?')}s<br>
<br>
<span style="color:#4A6080;">── PIPELINE STATUS ──</span><br>
{'<br>'.join(f"{'✓' if v else '✗'} {k.replace('_',' ').title()}" for k,v in pip.items())}<br>
<br>
<span style="color:#4A6080;">── PERFORMANCE ──</span><br>
Frames: {perf.get('frames_processed',0)}<br>
FPS (EMA): {perf.get('current_fps_ema',0):.1f}<br>
Lat (EMA): {perf.get('current_latency_ema',0):.1f} ms
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Diagnostics error: {e}")
            else:
                st.markdown(
                    '<div style="font-size:0.7rem;color:#4A6080;'
                    'font-family:monospace;">Engine not loaded</div>',
                    unsafe_allow_html=True,
                )

    return {
        "source_type":    source_type,
        "cam_index":      cam_index,
        "video_path":     video_path,
        "show_pose":      show_pose,
        "show_bbox":      show_bbox,
        "show_evidence":  show_evidence,
        "enable_alerts":  enable_alerts,
    }


# ═════════════════════════════════════════════════════════════════════════
# Monitoring controls
# ═════════════════════════════════════════════════════════════════════════

def _start_monitoring(source, engine) -> None:
    if engine is None:
        st.error("AI engine is not loaded. Cannot start monitoring.")
        return

    # Release previous camera if any
    _stop_monitoring()

    # Reset latched fall state & sound flag
    st.session_state.latched_fall = None
    st.session_state.fall_sound_triggered = False
    if engine is not None and hasattr(engine, "reset_safety"):
        engine.reset_safety()

    # Reset history for a fresh session timer and frame count
    if "history" in st.session_state and st.session_state.history is not None:
        st.session_state.history.reset()

    # Reset performance monitor for clean FPS calculation
    if hasattr(engine, "_perf"):
        from ai.inference import PerformanceMonitor
        engine._perf = PerformanceMonitor(
            warmup_frames=engine._cfg.get("performance", {}).get("warmup_frames", 5)
        )

    cam = CameraSource(source=source, loop_video=True)
    if not cam.open():
        st.error(f"Camera error: {cam.error}")
        return

    st.session_state.camera = cam
    st.session_state.monitoring = True
    st.session_state.video_path = source if isinstance(source, str) else None
    logger.info("Monitoring started. Source: %s", source)


def _stop_monitoring() -> None:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    st.session_state.monitoring = False
    st.session_state.latched_fall = None
    st.session_state.fall_sound_triggered = False
    engine = st.session_state.get("engine")
    if engine is not None and hasattr(engine, "reset_safety"):
        engine.reset_safety()
    logger.info("Monitoring stopped.")


def _reset_session() -> None:
    _stop_monitoring()
    st.session_state.history = SessionHistory()
    st.session_state.last_result = None
    st.session_state.latched_fall = None
    st.session_state.fall_sound_triggered = False
    engine = st.session_state.get("engine")
    if engine is not None and hasattr(engine, "reset_safety"):
        engine.reset_safety()
    logger.info("Session reset.")



# ═════════════════════════════════════════════════════════════════════════
# Main page render
# ═════════════════════════════════════════════════════════════════════════

def _render_main(settings: dict, engine) -> None:
    """Render all main-page components."""

    history: SessionHistory = st.session_state.history
    last_result = st.session_state.last_result

    # ── Extract data from last result ─────────────────────────────────────
    if last_result:
        sys_status  = last_result.get("system_status", "NORMAL")
        people      = last_result.get("people", [])
        metrics     = last_result.get("metrics", {})
        fps         = float(metrics.get("fps", 0.0))
        latency     = float(metrics.get("latency_ms", 0.0))
        people_cnt  = int(metrics.get("people_count", 0))
    else:
        sys_status  = "NORMAL"
        people      = []
        fps         = 0.0
        latency     = 0.0
        people_cnt  = 0

    if st.session_state.latched_fall:
        sys_status = "CRITICAL"

    # ── Header ────────────────────────────────────────────────────────────
    render_header()

    # ── Top System Status Cards ───────────────────────────────────────────
    top_cards_placeholder = st.empty()
    cam_open = st.session_state.camera is not None and st.session_state.camera.is_open()
    with top_cards_placeholder.container():
        render_top_system_cards(
            monitoring=st.session_state.monitoring,
            camera_connected=cam_open,
            classifier_mode=st.session_state.classifier_mode,
            system_status=sys_status,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── System status banner ──────────────────────────────────────────────
    system_status_placeholder = st.empty()
    with system_status_placeholder.container():
        render_system_status(sys_status)

    # ── Fall alerts (if any) ──────────────────────────────────────────────
    fall_alert_placeholder = st.empty()
    if settings.get("enable_alerts") and (people or st.session_state.latched_fall):
        with fall_alert_placeholder.container():
            render_fall_alert(people, latched_fall=st.session_state.latched_fall)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live video + metrics ──────────────────────────────────────────────
    col_video, col_side = st.columns([3, 1])

    with col_video:
        section("LIVE FEED")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col_side:
        section("LIVE METRICS")
        metrics_placeholder = st.empty()
        with metrics_placeholder.container():
            render_metric_cards(
                people_count=people_cnt,
                fps=fps,
                latency_ms=latency,
                alert_count=history.alert_count,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section("SESSION STATS")
        session_stats_placeholder = st.empty()
        with session_stats_placeholder.container():
            st.markdown(
                f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
            color:#8B9DC3;line-height:1.8;">
  <span style="color:#4A6080;">FRAMES:</span> {history.total_frames}<br>
  <span style="color:#4A6080;">AVG FPS:</span> {history.avg_fps:.1f}<br>
  <span style="color:#4A6080;">AVG LAT:</span> {history.avg_latency_ms:.0f} ms<br>
  <span style="color:#4A6080;">PEOPLE:</span> {history.people_observed}<br>
  <span style="color:#4A6080;">DURATION:</span> {history.session_duration_s:.0f} s
</div>
                """,
                unsafe_allow_html=True,
            )

    person_cards_placeholder = st.empty()
    evidence_placeholder = st.empty()

    # Placeholders for analytics & history table so they can update live
    section("ANALYTICS")
    acol1, acol2 = st.columns(2)
    with acol1:
        chart1_placeholder = st.empty()
    with acol2:
        chart2_placeholder = st.empty()

    chart3_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    section("ACTIVITY HISTORY")
    history_table_placeholder = st.empty()

    # Initial render of analytics containers when idle
    with chart1_placeholder.container():
        st.plotly_chart(
            activity_distribution_chart(history.activity_distribution),
            use_container_width=True,
        )
    with chart2_placeholder.container():
        all_records = history.get_recent_records(300)
        st.plotly_chart(
            activity_timeline_chart(all_records),
            use_container_width=True,
        )
    with chart3_placeholder.container():
        st.plotly_chart(
            performance_chart(all_records),
            use_container_width=True,
        )
    with history_table_placeholder.container():
        df = history.to_dataframe(n=100)
        if df.empty:
            st.markdown(
                '<div style="text-align:center;font-family:monospace;font-size:0.75rem;'
                'color:#4A6080;padding:16px;">No history yet. Start monitoring to record data.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=220,
            )

    # ── Activity frame processing ─────────────────────────────────────────
    if st.session_state.monitoring and st.session_state.camera is not None:
        _process_and_display(
            engine,
            video_placeholder,
            status_placeholder,
            metrics_placeholder,
            session_stats_placeholder,
            top_cards_placeholder,
            system_status_placeholder,
            fall_alert_placeholder,
            person_cards_placeholder,
            evidence_placeholder,
            chart1_placeholder,
            chart2_placeholder,
            chart3_placeholder,
            history_table_placeholder,
            settings,
        )
    else:
        # Show idle screen
        idle_img = _make_idle_frame()
        video_placeholder.image(
            idle_img, channels="BGR", use_container_width=True,
            caption="▶ Press START to begin monitoring"
        )
        if not st.session_state.monitoring:
            status_placeholder.markdown(
                '<div style="text-align:center;font-family:monospace;'
                'font-size:0.72rem;color:#4A6080;">MONITORING INACTIVE</div>',
                unsafe_allow_html=True,
            )

        if people:
            with person_cards_placeholder.container():
                cols_per_row = min(len(people), 3)
                section(f"ACTIVITY DETECTION  ·  {len(people)} PERSON(S) DETECTED")
                cols = st.columns(cols_per_row)
                for i, person in enumerate(people):
                    with cols[i % cols_per_row]:
                        render_person_card(person)

                if settings.get("show_evidence"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    section("WHY DID THE AI MAKE THIS DECISION?")
                    ev_cols = st.columns(min(len(people), 3))
                    for i, person in enumerate(people):
                        with ev_cols[i % len(ev_cols)]:
                            evidence = person.get("evidence", [])
                            pid = person.get("person_id", "?")
                            activity = person.get("activity", "UNKNOWN")
                            ev_html = "".join(
                                f'<div style="font-size:0.78rem;color:#8B9DC3;padding:3px 0;">'
                                f'✓ {e}</div>'
                                for e in evidence
                            ) or '<div style="font-size:0.78rem;color:#4A6080;">No evidence available</div>'
                            st.markdown(
                                f'<div style="background:#0D1626;border:1px solid #1E2D3D;'
                                f'border-radius:6px;padding:14px;">'
                                f'<div style="font-family:monospace;font-size:0.65rem;'
                                f'color:#4A6080;margin-bottom:8px;letter-spacing:0.1em;">'
                                f'PERSON #{pid} — {activity}</div>'
                                f'{ev_html}</div>',
                                unsafe_allow_html=True,
                            )
        else:
            with person_cards_placeholder.container():
                section("ACTIVITY DETECTION")
                st.markdown(
                    '<div style="text-align:center;font-family:monospace;font-size:0.75rem;'
                    'color:#4A6080;padding:20px;">No persons detected in current frame</div>',
                    unsafe_allow_html=True,
                )

    render_footer()


# ═════════════════════════════════════════════════════════════════════════
# Core processing loop (continuous streaming)
# ═════════════════════════════════════════════════════════════════════════

def _process_and_display(
    engine,
    video_placeholder,
    status_placeholder,
    metrics_placeholder,
    session_stats_placeholder,
    top_cards_placeholder,
    system_status_placeholder,
    fall_alert_placeholder,
    person_cards_placeholder,
    evidence_placeholder,
    chart1_placeholder,
    chart2_placeholder,
    chart3_placeholder,
    history_table_placeholder,
    settings: dict,
) -> None:
    """Read frames continuously in a real-time loop, run AI pipeline, update UI elements."""
    cam: CameraSource = st.session_state.camera

    if cam is None or not cam.is_open():
        status_placeholder.error("Camera source is not open. Click START to begin monitoring.")
        _stop_monitoring()
        return

    history: SessionHistory = st.session_state.history
    frame_counter = 0

    while st.session_state.monitoring and cam.is_open():
        ok, frame = cam.read()
        if not ok or frame is None:
            status_placeholder.warning("End of video stream or camera disconnected.")
            _stop_monitoring()
            break

        # Check for dark / black frame from camera hardware
        if frame.mean() < 0.5:
            status_placeholder.markdown(
                '<div style="background:#1E1400;border:1px solid #D4A017;padding:8px;'
                'border-radius:4px;font-family:monospace;font-size:0.75rem;color:#F5A623;">'
                '⚠ Camera feed is dark/black (hardware shutter or privacy mode). '
                'Switch source to "Demo Dataset Video" in sidebar if dark.</div>',
                unsafe_allow_html=True,
            )

        # Run AI engine
        try:
            result, annotated_frame = engine.process_frame(frame)
        except Exception as exc:
            logger.error("process_frame error: %s", exc)
            status_placeholder.error(f"AI pipeline error: {exc}")
            annotated_frame = frame
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "system_status": "NORMAL",
                "people": [],
                "metrics": {"people_count": 0, "fps": 0, "latency_ms": 0},
            }

        # Check for latched fall event
        latched_fall = result.get("latched_fall") or st.session_state.latched_fall
        if latched_fall:
            st.session_state.latched_fall = latched_fall
            if not st.session_state.get("fall_sound_triggered", False):
                st.session_state.fall_sound_triggered = True
                st.toast("🚨 FALL DETECTED — CRITICAL SAFETY ALERT!", icon="🚨")
                try:
                    from dashboard.ui import generate_warning_beep_wav_b64
                    sound_b64 = generate_warning_beep_wav_b64()
                    st.markdown(
                        f'<audio autoplay style="display:none;"><source src="{sound_b64}" type="audio/wav"></audio>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    logger.warning("Could not play audio alert: %s", e)

        # Store result in session state & history
        st.session_state.last_result = result
        history.record(result)

        # Display annotated frame (BGR -> RGB)
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_frame, use_container_width=True)

        sys_status = result.get("system_status", "NORMAL")
        if st.session_state.latched_fall:
            sys_status = "CRITICAL"

        people = result.get("people", [])

        with top_cards_placeholder.container():
            render_top_system_cards(
                monitoring=True,
                camera_connected=True,
                classifier_mode=st.session_state.classifier_mode,
                system_status=sys_status,
            )

        with system_status_placeholder.container():
            render_system_status(sys_status)

        # Update fall alert panel
        if settings.get("enable_alerts") and (people or st.session_state.latched_fall):
            with fall_alert_placeholder.container():
                render_fall_alert(people, latched_fall=st.session_state.latched_fall)
        else:
            fall_alert_placeholder.empty()

        # Update live metric cards container
        metrics = result.get("metrics", {})
        live_fps = float(metrics.get("fps", 0.0))
        live_lat = float(metrics.get("latency_ms", 0.0))
        live_cnt = int(metrics.get("people_count", 0))

        with metrics_placeholder.container():
            render_metric_cards(
                people_count=live_cnt,
                fps=live_fps,
                latency_ms=live_lat,
                alert_count=history.alert_count,
            )

        with session_stats_placeholder.container():
            st.markdown(
                f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
            color:#8B9DC3;line-height:1.8;">
  <span style="color:#4A6080;">FRAMES:</span> {history.total_frames}<br>
  <span style="color:#4A6080;">AVG FPS:</span> {history.avg_fps:.1f}<br>
  <span style="color:#4A6080;">AVG LAT:</span> {history.avg_latency_ms:.0f} ms<br>
  <span style="color:#4A6080;">PEOPLE:</span> {history.people_observed}<br>
  <span style="color:#4A6080;">DURATION:</span> {history.session_duration_s:.0f} s
</div>
                """,
                unsafe_allow_html=True,
            )

        # Update person detection cards & evidence
        if people:
            with person_cards_placeholder.container():
                cols_per_row = min(len(people), 3)
                section(f"ACTIVITY DETECTION  ·  {len(people)} PERSON(S) DETECTED")
                cols = st.columns(cols_per_row)
                for i, person in enumerate(people):
                    with cols[i % cols_per_row]:
                        render_person_card(person)

            if settings.get("show_evidence"):
                with evidence_placeholder.container():
                    st.markdown("<br>", unsafe_allow_html=True)
                    section("WHY DID THE AI MAKE THIS DECISION?")
                    ev_cols = st.columns(min(len(people), 3))
                    for i, person in enumerate(people):
                        with ev_cols[i % len(ev_cols)]:
                            evidence = person.get("evidence", [])
                            pid = person.get("person_id", "?")
                            activity = person.get("activity", "UNKNOWN")
                            ev_html = "".join(
                                f'<div style="font-size:0.78rem;color:#8B9DC3;padding:3px 0;">'
                                f'✓ {e}</div>'
                                for e in evidence
                            ) or '<div style="font-size:0.78rem;color:#4A6080;">No evidence available</div>'
                            st.markdown(
                                f'<div style="background:#0D1626;border:1px solid #1E2D3D;'
                                f'border-radius:6px;padding:14px;">'
                                f'<div style="font-family:monospace;font-size:0.65rem;'
                                f'color:#4A6080;margin-bottom:8px;letter-spacing:0.1em;">'
                                f'PERSON #{pid} — {activity}</div>'
                                f'{ev_html}</div>',
                                unsafe_allow_html=True,
                            )
            else:
                evidence_placeholder.empty()
        else:
            with person_cards_placeholder.container():
                section("ACTIVITY DETECTION")
                st.markdown(
                    '<div style="text-align:center;font-family:monospace;font-size:0.75rem;'
                    'color:#4A6080;padding:20px;">No persons detected in current frame</div>',
                    unsafe_allow_html=True,
                )
            evidence_placeholder.empty()

        # Update real-time analytics graphs & history table at practical intervals (every 5 frames)
        frame_counter += 1
        if frame_counter % 5 == 0 or frame_counter == 1:
            recent_records = history.get_recent_records(300)
            with chart1_placeholder.container():
                st.plotly_chart(
                    activity_distribution_chart(history.activity_distribution),
                    use_container_width=True,
                )
            with chart2_placeholder.container():
                st.plotly_chart(
                    activity_timeline_chart(recent_records),
                    use_container_width=True,
                )
            with chart3_placeholder.container():
                st.plotly_chart(
                    performance_chart(recent_records),
                    use_container_width=True,
                )
            with history_table_placeholder.container():
                df = history.to_dataframe(n=100)
                if df.empty:
                    st.markdown(
                        '<div style="text-align:center;font-family:monospace;font-size:0.75rem;'
                        'color:#4A6080;padding:16px;">No history yet. Start monitoring to record data.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=220,
                    )

        time.sleep(0.005)

    # Final analytics refresh when stream stops
    recent_records = history.get_recent_records(300)
    with chart1_placeholder.container():
        st.plotly_chart(
            activity_distribution_chart(history.activity_distribution),
            use_container_width=True,
        )
    with chart2_placeholder.container():
        st.plotly_chart(
            activity_timeline_chart(recent_records),
            use_container_width=True,
        )
    with chart3_placeholder.container():
        st.plotly_chart(
            performance_chart(recent_records),
            use_container_width=True,
        )
    with history_table_placeholder.container():
        df = history.to_dataframe(n=100)
        if not df.empty:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=220,
            )



# ═════════════════════════════════════════════════════════════════════════
# Idle frame
# ═════════════════════════════════════════════════════════════════════════

def _make_idle_frame() -> np.ndarray:
    """Generate a dark idle frame with text overlay."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (10, 14, 26)   # dark navy

    # Grid lines (subtle)
    for x in range(0, w, 80):
        cv2.line(img, (x, 0), (x, h), (25, 40, 60), 1)
    for y in range(0, h, 60):
        cv2.line(img, (0, y), (w, y), (25, 40, 60), 1)

    # Centre text
    cv2.putText(
        img, "MONITORING INACTIVE",
        (130, h // 2 - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (45, 70, 100), 1,
    )
    cv2.putText(
        img, "Press START to begin",
        (185, h // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 60, 80), 1,
    )
    return img


# ═════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Load engine once
    engine = _load_engine()

    # Show engine error if any
    if st.session_state.engine_error:
        st.error(
            f"⚠ AI Engine Error: {st.session_state.engine_error}\n\n"
            "Ensure `yolo11n-pose.pt` is present or internet is available for download. "
            "Check that all dependencies are installed: `pip install -r requirements.txt`"
        )

    # Sidebar
    settings = _render_sidebar(engine)

    # Main page
    _render_main(settings, engine)


if __name__ == "__main__":
    main()
