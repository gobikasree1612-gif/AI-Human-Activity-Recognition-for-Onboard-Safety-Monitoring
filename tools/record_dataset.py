"""
record_dataset.py
=================
Webcam Dataset Recording Utility for onboard-har.

Enables controlled recording of real training video clips directly into
the structured `dataset/raw/<CLASS>/` hierarchy.

Features:
- Live camera preview with on-screen recording timer and metadata overlay
- Interactive start/stop triggers (SPACE bar)
- Configurable FPS, resolution, and camera device index
- Automatic collision-free filename generation
- Direct manifest updating
- Prominent safety warning for staged FALLING demonstrations
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import sys
import time
from typing import Optional

import cv2

# Project root in path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.schemas import Activity
from dashboard.camera import open_camera_capture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("record_dataset")

VALID_CLASSES = Activity.ALL


def update_manifest_file(manifest_path: str, record: dict) -> None:
    """Append a newly recorded clip to the dataset manifest."""
    headers = [
        "video_id",
        "filename",
        "class",
        "subject_id",
        "session_id",
        "camera_id",
        "environment",
        "resolution",
        "fps",
        "duration_seconds",
        "lighting",
        "camera_angle",
        "distance",
        "notes",
    ]
    file_exists = os.path.isfile(manifest_path)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        clean_row = {h: record.get(h, "") for h in headers}
        writer.writerow(clean_row)
    logger.info("Manifest updated: %s", manifest_path)


def record_session(
    activity_class: str,
    subject_id: str,
    session_id: str,
    camera_index: int = 0,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    environment: str = "lab",
    lighting: str = "standard",
    camera_angle: str = "eye_level",
    distance: str = "3m",
    output_root: str = "dataset/raw",
    manifest_path: str = "dataset/manifest.csv",
    target_duration: float = 0.0,
    headless_frames: int = 0,  # >0 for automated tests without GUI
) -> Optional[str]:
    """
    Run an interactive recording session with webcam preview.
    Returns path to saved video, or None if aborted.
    """
    normalized_class = activity_class.strip().upper()
    if normalized_class not in VALID_CLASSES:
        logger.error(
            "Invalid class '%s'. Must be one of: %s",
            activity_class,
            ", ".join(VALID_CLASSES),
        )
        return None

    # Safety warning for FALLING
    if normalized_class == Activity.FALLING:
        print("\n" + "=" * 60)
        print("WARNING:")
        print("FALLING DATA MUST BE STAGED SAFELY.\n")
        print("Use appropriate padding/crash protection.")
        print("Do not perform dangerous falls.")
        print("Stop immediately if unsafe.")
        print("=" * 60 + "\n")

    # Camera initialization
    logger.info("Opening camera device %d...", camera_index)
    cap, open_err = open_camera_capture(
        camera_index,
        width=width,
        height=height,
        fps=fps,
    )
    if cap is None or not cap.isOpened():
        logger.error(
            "Failed to open camera index %d (%s). Verify camera connection.",
            camera_index,
            open_err or "Unknown error",
        )
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fps = float(raw_fps) if (raw_fps is not None and raw_fps > 0) else float(fps)

    # Setup directories
    target_dir = os.path.join(output_root, normalized_class)
    os.makedirs(target_dir, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_stem = f"{normalized_class}_{subject_id}_{session_id}_{timestamp_str}"
    out_path = os.path.join(target_dir, f"{base_stem}.mp4")

    # Prevent accidental overwrite
    counter = 1
    while os.path.exists(out_path):
        out_path = os.path.join(target_dir, f"{base_stem}_{counter}.mp4")
        counter += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: Optional[cv2.VideoWriter] = None

    is_recording = False
    start_time = 0.0
    last_toggle_time = 0.0
    recorded_frames = 0
    saved_file: Optional[str] = None
    termination_reason = "Aborted or cancelled before completion"
    consecutive_read_failures = 0
    max_consecutive_failures = 30

    logger.info("Starting preview. Target duration: %.1fs. Press [Q] to quit.", target_duration)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                logger.warning(
                    "Camera frame read failure (%d/%d) at frame_idx=%d",
                    consecutive_read_failures,
                    max_consecutive_failures,
                    frame_idx,
                )
                if consecutive_read_failures < max_consecutive_failures:
                    time.sleep(0.01)
                    continue
                else:
                    termination_reason = f"Camera read failure: {consecutive_read_failures} consecutive frame read drops"
                    logger.error("Recording failed: %s", termination_reason)
                    break
            else:
                consecutive_read_failures = 0

            frame_idx += 1
            h, w = frame.shape[:2]

            # For automated headless testing
            if headless_frames > 0:
                if frame_idx == 1:
                    is_recording = True
                    start_time = time.monotonic()
                    last_toggle_time = start_time
                    writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (w, h))
                    if not writer.isOpened():
                        termination_reason = "Failed to initialize VideoWriter"
                        logger.error("Recording failed: %s", termination_reason)
                        break
                if is_recording and writer is not None:
                    writer.write(frame)
                    recorded_frames += 1
                if frame_idx >= headless_frames:
                    termination_reason = f"Headless frame limit ({headless_frames} frames) reached"
                    saved_file = out_path
                    break
                continue

            # Auto-start if target_duration > 0 and not yet recording
            if target_duration > 0 and not is_recording and writer is None:
                is_recording = True
                start_time = time.monotonic()
                last_toggle_time = start_time
                recorded_frames = 0
                writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (w, h))
                if not writer.isOpened():
                    termination_reason = f"Failed to initialize VideoWriter at {out_path}"
                    logger.error("Recording failed: %s", termination_reason)
                    break
                logger.info("Auto-recording STARTED for %.1fs -> %s", target_duration, out_path)

            # Auto-stop if target_duration is reached
            if is_recording and target_duration > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= target_duration:
                    is_recording = False
                    if writer:
                        writer.release()
                        writer = None
                    termination_reason = f"Target duration {target_duration:.1f}s reached normally"
                    logger.info("Recording completed normally: %s. Recorded %d frames in %.2fs.", termination_reason, recorded_frames, elapsed)
                    saved_file = out_path
                    break

            display_frame = frame.copy()

            # Overlay information & frame writing with real-time frame rate synchronization
            if not is_recording:
                status_text = "STANDBY - Press [SPACE] to Record" if target_duration == 0 else "STANDBY"
                status_color = (0, 255, 255)  # Yellow
                dur_text = "0.0s"
            else:
                elapsed = time.monotonic() - start_time
                target_str = f" / {target_duration:.1f}s" if target_duration > 0 else ""
                if target_duration == 0:
                    status_text = "● RECORDING - Press [SPACE] to Stop"
                else:
                    status_text = f"● RECORDING - Auto ({target_duration:.1f}s)"
                status_color = (0, 0, 255)  # Red
                dur_text = f"{elapsed:.1f}s{target_str} ({recorded_frames} frames)"

                if writer is not None:
                    # Sync written frames with elapsed time at target_fps
                    expected_frames = max(1, int(round(elapsed * actual_fps)))
                    frames_to_add = max(1, expected_frames - recorded_frames)
                    for _ in range(frames_to_add):
                        writer.write(frame)
                        recorded_frames += 1

            # Draw HUD
            cv2.rectangle(display_frame, (10, 10), (w - 10, 75), (20, 20, 20), -1)
            cv2.putText(display_frame, f"CLASS: {normalized_class} | SUBJ: {subject_id} | SESS: {session_id}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.putText(display_frame, f"{status_text} | {dur_text}",
                        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            if normalized_class == Activity.FALLING:
                cv2.putText(display_frame, "CAUTION: SAFE STAGED ONLY", (w - 280, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            try:
                cv2.imshow("Onboard-HAR Dataset Recorder", display_frame)
                key = cv2.waitKey(1) & 0xFF
            except Exception:
                # If OpenCV GUI is headless/disabled (e.g., in unit tests)
                key = -1

            if key == 32:  # SPACE bar
                if target_duration == 0:
                    now = time.monotonic()
                    if now - last_toggle_time >= 0.5:
                        last_toggle_time = now
                        if not is_recording:
                            # Start recording
                            is_recording = True
                            start_time = time.monotonic()
                            recorded_frames = 0
                            writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (w, h))
                            if not writer.isOpened():
                                termination_reason = "Failed to initialize VideoWriter on manual start"
                                logger.error("Recording failed: %s", termination_reason)
                                break
                            logger.info("Recording STARTED -> %s", out_path)
                        else:
                            # Stop recording
                            is_recording = False
                            if writer:
                                writer.release()
                                writer = None
                            elapsed = time.monotonic() - start_time
                            termination_reason = f"Manual stop by user after {elapsed:.1f}s"
                            logger.info("Recording completed normally: %s. Captured %d frames.", termination_reason, recorded_frames)
                            saved_file = out_path
                            break
                else:
                    logger.debug("SPACE key ignored during timed recording mode (duration=%.1fs)", target_duration)

            elif key in (ord("q"), ord("Q"), 27):  # Q or ESC
                termination_reason = "Cancelled by user (Q/ESC key pressed)"
                logger.info("Recording cancelled: %s", termination_reason)
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Clean up incomplete file if cancelled
    if saved_file is None and os.path.isfile(out_path) and headless_frames == 0:
        try:
            os.remove(out_path)
        except Exception:
            pass

    if headless_frames > 0 and recorded_frames > 0:
        saved_file = out_path

    if saved_file and os.path.isfile(saved_file) and recorded_frames > 0:
        duration_sec = round(recorded_frames / actual_fps, 2)
        rel_filename = os.path.relpath(saved_file, output_root).replace("\\", "/")
        rec_meta = {
            "video_id": os.path.splitext(os.path.basename(saved_file))[0],
            "filename": rel_filename,
            "class": normalized_class,
            "subject_id": subject_id,
            "session_id": session_id,
            "camera_id": f"cam_{camera_index}",
            "environment": "lab",
            "resolution": f"{actual_w}x{actual_h}",
            "fps": str(round(actual_fps, 2)),
            "duration_seconds": str(duration_sec),
            "lighting": lighting,
            "camera_angle": camera_angle,
            "distance": distance,
            "notes": "recorded_via_webcam_tool",
        }
        update_manifest_file(manifest_path, rec_meta)
        print("\n" + "=" * 60)
        print("RECORDING CONFIRMATION")
        print("=" * 60)
        print(f"Status:             Recording completed normally")
        print(f"Termination reason: {termination_reason}")
        print(f"Saved file:         {saved_file}")
        print(f"Activity class:     {normalized_class}")
        print(f"Subject ID:         {subject_id}")
        print(f"Session ID:         {session_id}")
        print(f"Frames captured:    {recorded_frames}")
        print(f"Duration:           {duration_sec:.1f}s")
        print(f"Resolution / FPS:   {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        print(f"Manifest entry:     Updated in {manifest_path}")
        print("=" * 60 + "\n")
        logger.info("Successfully recorded video: %s (%.1fs, %d frames). Reason: %s", saved_file, duration_sec, recorded_frames, termination_reason)
        return saved_file

    logger.warning("Recording finished without saving file. Termination reason: %s", termination_reason)
    print("\n" + "=" * 60)
    print("RECORDING TERMINATED WITHOUT SAVING")
    print("=" * 60)
    print(f"Reason: {termination_reason}")
    print("=" * 60 + "\n")
    return None


def run_guided_session(
    subject_id: str,
    session_id: str,
    camera_index: int = 0,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    output_root: str = "dataset/raw",
    manifest_path: str = "dataset/manifest.csv",
    classes_to_record: Optional[List[str]] = None,
    interactive_prompt: bool = True,
    target_duration: float = 10.0,
    headless_frames: int = 0,
) -> List[str]:
    """
    Run an interactive guided dataset recording sequence through all target activity classes.

    Sequence: STANDING -> SITTING -> WALKING -> RUNNING -> FALLING
    """
    classes = classes_to_record or list(VALID_CLASSES)
    saved_files = []

    print("\n" + "=" * 65)
    print("GUIDED REAL DATA COLLECTION SESSION")
    print("=" * 65)
    print(f"Subject: {subject_id} | Session: {session_id}")
    print("Sequence: " + " -> ".join(classes))
    print("=" * 65 + "\n")

    for cls_name in classes:
        while True:
            print(f"\n--- NEXT CLASS: {cls_name} ---")
            if cls_name == Activity.FALLING:
                print("SAFETY WARNING: Ensure protective crash mat / padding is in place!")
                print("Only perform safe, staged fall demonstrations.")

            if interactive_prompt:
                try:
                    ans = input(f"Ready to record {cls_name}? ([y]=start, [s]=skip, [q]=quit): ").strip().lower()
                except EOFError:
                    ans = "s"

                if ans == "s":
                    print(f"Skipping class {cls_name}.")
                    break
                elif ans == "q":
                    print("Ending guided recording session.")
                    return saved_files
                elif ans not in ("y", "", "yes"):
                    print("Invalid choice. Try 'y', 's', or 'q'.")
                    continue

            result = record_session(
                activity_class=cls_name,
                subject_id=subject_id,
                session_id=session_id,
                camera_index=camera_index,
                fps=fps,
                width=width,
                height=height,
                output_root=output_root,
                manifest_path=manifest_path,
                target_duration=target_duration,
                headless_frames=headless_frames,
            )

            if result and os.path.isfile(result):
                saved_files.append(result)
                print(f"SUCCESS: Saved {cls_name} clip -> {result}")
            else:
                print(f"FAILED / CANCELLED: No video file saved for class {cls_name}.")

            if interactive_prompt:
                try:
                    repeat_ans = input(f"Record ANOTHER clip for {cls_name}? ([y]=yes, [n]=next class): ").strip().lower()
                except EOFError:
                    repeat_ans = "n"

                if repeat_ans in ("y", "yes"):
                    continue
            break

    print("\n" + "=" * 65)
    print("GUIDED SESSION COMPLETE")
    print(f"Total clips recorded: {len(saved_files)}")
    print("=" * 65 + "\n")
    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Record real activity video clips using webcam.")
    parser.add_argument("--guided", action="store_true", help="Launch interactive guided recording sequence through all 5 activity classes.")
    parser.add_argument("--class", dest="activity_class", choices=VALID_CLASSES, default=None,
                        help="Activity class to record (Required unless --guided is passed)")
    parser.add_argument("--subject", default="S01", help="Subject identifier (e.g. S01)")
    parser.add_argument("--session", default="SESSION01", help="Session identifier (e.g. SESSION01)")
    parser.add_argument("--duration", type=float, default=0.0, help="Target recording duration in seconds (0 = manual stop)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording FPS")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--environment", default="lab", help="Environment tag (default: lab)")
    parser.add_argument("--lighting", default="standard", help="Lighting condition (default: standard)")
    parser.add_argument("--camera-angle", default="eye_level", help="Camera angle (default: eye_level)")
    parser.add_argument("--distance", default="3m", help="Subject distance (default: 3m)")
    parser.add_argument("--output-dir", default="dataset/raw", help="Target raw dataset directory")
    parser.add_argument("--manifest", default="dataset/manifest.csv", help="Path to manifest.csv")
    args = parser.parse_args()

    if args.guided:
        run_guided_session(
            subject_id=args.subject,
            session_id=args.session,
            camera_index=args.cam,
            fps=args.fps,
            width=args.width,
            height=args.height,
            output_root=args.output_dir,
            manifest_path=args.manifest,
            target_duration=args.duration or 10.0,
        )
    else:
        if not args.activity_class:
            parser.error("argument --class is required unless --guided is specified.")

        record_session(
            activity_class=args.activity_class,
            subject_id=args.subject,
            session_id=args.session,
            camera_index=args.cam,
            fps=args.fps,
            width=args.width,
            height=args.height,
            environment=args.environment,
            lighting=args.lighting,
            camera_angle=args.camera_angle,
            distance=args.distance,
            output_root=args.output_dir,
            manifest_path=args.manifest,
            target_duration=args.duration,
        )


if __name__ == "__main__":
    main()
