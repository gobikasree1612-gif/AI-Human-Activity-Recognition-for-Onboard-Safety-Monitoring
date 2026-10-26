"""
dashboard/analytics.py
=======================
Plotly chart builders for the analytics panel.

All charts are built from real session history data.
No fake/hardcoded values.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette (aerospace dark theme)
# ---------------------------------------------------------------------------

ACTIVITY_COLORS = {
    "STANDING": "#00C896",   # teal-green
    "SITTING":  "#4A9EFF",   # blue
    "WALKING":  "#F5A623",   # amber
    "RUNNING":  "#E05CF5",   # purple
    "FALLING":  "#FF3B3B",   # red
    "UNKNOWN":  "#6B7280",   # grey
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,30,0.8)",
    font=dict(color="#C9D1D9", family="monospace", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ---------------------------------------------------------------------------
# Activity distribution bar chart
# ---------------------------------------------------------------------------

def activity_distribution_chart(activity_counts: Dict[str, int]) -> go.Figure:
    """
    Horizontal bar chart of activity counts from the current session.

    Parameters
    ----------
    activity_counts : dict
        e.g. {"STANDING": 120, "WALKING": 85, "FALLING": 3}
    """
    if not activity_counts:
        fig = go.Figure()
        fig.update_layout(
            **LAYOUT_DEFAULTS,
            title=dict(text="No activity data yet", font=dict(color="#6B7280")),
        )
        return fig

    activities = list(activity_counts.keys())
    counts = list(activity_counts.values())
    colors = [ACTIVITY_COLORS.get(a, "#6B7280") for a in activities]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=activities,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=counts,
            textposition="outside",
            textfont=dict(color="#C9D1D9", size=11),
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Activity Distribution", font=dict(color="#8B9DC3", size=13)),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
            tickfont=dict(color="#8B9DC3"),
        ),
        yaxis=dict(
            tickfont=dict(color="#C9D1D9", size=12),
            categoryorder="total ascending",
        ),
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Activity timeline scatter
# ---------------------------------------------------------------------------

def activity_timeline_chart(records) -> go.Figure:
    """
    Scatter chart showing activity over time for each person.

    Parameters
    ----------
    records : list of HistoryRecord
    """
    if not records:
        fig = go.Figure()
        fig.update_layout(
            **LAYOUT_DEFAULTS,
            title=dict(text="No timeline data yet", font=dict(color="#6B7280")),
        )
        return fig

    import datetime

    rows = []
    for r in records[-300:]:
        try:
            ts = datetime.datetime.fromisoformat(r.timestamp)
        except Exception:
            continue
        rows.append(
            {
                "time": ts,
                "activity": r.activity,
                "person": f"#{r.person_id}",
                "confidence": round(r.confidence * 100, 1),
            }
        )

    if not rows:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    df = pd.DataFrame(rows)

    fig = go.Figure()
    for activity in df["activity"].unique():
        mask = df["activity"] == activity
        sub = df[mask]
        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub["person"],
                mode="markers",
                name=activity,
                marker=dict(
                    color=ACTIVITY_COLORS.get(activity, "#6B7280"),
                    size=8,
                    symbol="circle",
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Activity: {activity}<br>"
                    "Time: %{x}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Activity Timeline", font=dict(color="#8B9DC3", size=13)),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            tickfont=dict(color="#8B9DC3"),
            title=None,
        ),
        yaxis=dict(
            tickfont=dict(color="#C9D1D9"),
            title=None,
        ),
        legend=dict(
            font=dict(color="#C9D1D9", size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# FPS / latency over time
# ---------------------------------------------------------------------------

def performance_chart(records) -> go.Figure:
    """Line chart of FPS and latency sampled from history records."""
    if not records:
        fig = go.Figure()
        fig.update_layout(
            **LAYOUT_DEFAULTS,
            title=dict(text="No performance data yet", font=dict(color="#6B7280")),
        )
        return fig

    import datetime

    # Sample every N records to keep chart light
    sample_records = records[-200:]
    rows = []
    for r in sample_records:
        try:
            ts = datetime.datetime.fromisoformat(r.timestamp)
        except Exception:
            continue
        rows.append({"time": ts, "fps": r.fps, "latency": r.latency_ms})

    if not rows:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["fps"],
            mode="lines",
            name="FPS",
            line=dict(color="#00C896", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,200,150,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["latency"],
            mode="lines",
            name="Latency (ms)",
            line=dict(color="#F5A623", width=1.5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Performance Over Time", font=dict(color="#8B9DC3", size=13)),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            tickfont=dict(color="#8B9DC3"),
        ),
        yaxis=dict(
            title=dict(text="FPS", font=dict(color="#00C896")),
            tickfont=dict(color="#00C896"),
        ),
        yaxis2=dict(
            title=dict(text="Latency (ms)", font=dict(color="#F5A623")),
            overlaying="y",
            side="right",
            tickfont=dict(color="#F5A623"),
        ),
        legend=dict(
            font=dict(color="#C9D1D9", size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
        height=200,
    )
    return fig
