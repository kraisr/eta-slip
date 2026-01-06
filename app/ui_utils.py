from __future__ import annotations


def risk_color(p: float, thr: float) -> str:
    if p >= 0.75:
        return "#d73027"  # red
    if p >= 0.50:
        return "#fc8d59"  # orange
    if p >= 0.25:
        return "#fee08b"  # yellow
    return "#1a9850"  # green


def bar_html(p: float, thr: float) -> str:
    pct = max(0.0, min(1.0, float(p))) * 100.0
    color = risk_color(float(p), thr)
    return f"""
    <div style="width: 100%; background: rgba(0,0,0,0.06); border-radius: 10px; height: 12px; overflow: hidden;">
      <div style="width: {pct:.1f}%; background: {color}; height: 12px;"></div>
    </div>
    """
