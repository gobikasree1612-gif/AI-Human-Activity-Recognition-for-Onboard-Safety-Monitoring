# Deployment Guide

This guide outlines requirements and procedures for deploying **onboard-har** in real-time edge environments.

## Edge System Requirements

- **Processor:** Dual-core x86_64 or ARM64 (e.g. Raspberry Pi 4/5, Jetson Nano/Orin, laptop/desktop CPU).
- **RAM:** Minimum 2 GB (4 GB recommended).
- **Camera:** Standard USB / CSI webcam or RTSP IP camera stream ($640 \times 480$ or $1280 \times 720$ resolution @ 20+ FPS).
- **OS:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS.

## Quick Start Deployment

1. **Clone repository & install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run System Health Check:**
   ```bash
   python final_system_check.py
   ```
3. **Launch Real-Time Dashboard:**
   ```bash
   streamlit run app.py
   ```
4. **Run AI Engine directly on Video File or Webcam:**
   ```bash
   python demo_ai.py --source 0
   ```
