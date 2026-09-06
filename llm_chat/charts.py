"""Chart generation module using matplotlib."""

import json
import logging
import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("generated_charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Charts older than this are cleaned up automatically
CHART_MAX_AGE_SECONDS = 3600

# Dark colour palette that looks good on a dark chat UI
PALETTE = [
    "#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8",
    "#4dd0e1", "#fff176", "#f06292", "#aed581", "#90a4ae",
]


def cleanup_old_charts() -> None:
    """Delete chart images older than CHART_MAX_AGE_SECONDS."""
    now = time.time()
    for f in CHARTS_DIR.glob("*.png"):
        try:
            if now - f.stat().st_mtime > CHART_MAX_AGE_SECONDS:
                f.unlink()
                logger.debug("Cleaned up old chart: %s", f.name)
        except OSError:
            pass


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply a consistent dark theme to a figure and axes."""
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#282840")
    ax.tick_params(colors="#cdd6f4", labelsize=9)
    ax.xaxis.label.set_color("#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.title.set_color("#cdd6f4")
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    ax.grid(axis="y", color="#45475a", alpha=0.3, linestyle="--")


def _coerce_values(raw: list) -> list[float]:
    """Best-effort conversion of a list to floats."""
    out = []
    for v in raw:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def generate_chart(chart_spec: dict) -> str:
    """Generate a PNG chart from *chart_spec* and return the filename.

    Supported keys in *chart_spec*:
        type        – "bar" | "horizontal_bar" | "line" | "pie"
        title       – chart title (str)
        labels      – list of category / x-axis labels
        values      – list of numeric values (single series)
        datasets    – list of {"label": str, "values": [...]} (multi-series)
        x_label     – optional x-axis label
        y_label     – optional y-axis label
    """
    cleanup_old_charts()

    chart_id = uuid.uuid4().hex[:12]
    filename = f"{chart_id}.png"
    filepath = CHARTS_DIR / filename

    chart_type = chart_spec.get("type", "bar")
    title = chart_spec.get("title", "")
    labels = chart_spec.get("labels", [])
    values = _coerce_values(chart_spec.get("values", []))
    datasets = chart_spec.get("datasets", [])
    x_label = chart_spec.get("x_label", "")
    y_label = chart_spec.get("y_label", "")

    # Basic validation
    if not labels:
        raise ValueError("Chart spec must include non-empty 'labels'")
    if not values and not datasets:
        raise ValueError("Chart spec must include 'values' or 'datasets'")

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_theme(fig, ax)

    # ── Bar chart ────────────────────────────────────────────────────
    if chart_type == "bar":
        if datasets:
            x = np.arange(len(labels))
            n = len(datasets)
            width = 0.8 / n
            for i, ds in enumerate(datasets):
                offset = (i - n / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    _coerce_values(ds.get("values", [])),
                    width,
                    label=ds.get("label", f"Series {i + 1}"),
                    color=PALETTE[i % len(PALETTE)],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend(
                facecolor="#282840", edgecolor="#45475a", labelcolor="#cdd6f4"
            )
        else:
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
            ax.bar(labels, values, color=colors)
            ax.set_xticklabels(labels, rotation=45, ha="right")

    # ── Horizontal bar chart ─────────────────────────────────────────
    elif chart_type == "horizontal_bar":
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
        ax.barh(labels, values, color=colors)
        if values:
            pad = max(values) * 0.02
            for i, v in enumerate(values):
                ax.text(
                    v + pad, i, f"{v:,.1f}",
                    va="center", color="#cdd6f4", fontsize=9,
                )

    # ── Line chart ───────────────────────────────────────────────────
    elif chart_type == "line":
        if datasets:
            for i, ds in enumerate(datasets):
                ax.plot(
                    labels,
                    _coerce_values(ds.get("values", [])),
                    marker="o", markersize=4, linewidth=2,
                    label=ds.get("label", f"Series {i + 1}"),
                    color=PALETTE[i % len(PALETTE)],
                )
            ax.legend(
                facecolor="#282840", edgecolor="#45475a", labelcolor="#cdd6f4"
            )
        else:
            ax.plot(
                labels, values,
                marker="o", markersize=4, linewidth=2,
                color=PALETTE[0],
            )
        if len(labels) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # ── Pie chart ────────────────────────────────────────────────────
    elif chart_type == "pie":
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=colors, textprops={"color": "#cdd6f4"},
        )
        for at in autotexts:
            at.set_color("#1e1e2e")
            at.set_fontweight("bold")
        ax.grid(False)

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10)

    plt.tight_layout()
    fig.savefig(
        filepath, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    plt.close(fig)

    logger.info("Generated chart: %s (type=%s)", filename, chart_type)
    return filename