"""Draw one figure for a mission run: top-down path, altitude per phase, raven detections."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STATE_COLORS = {
    "idle": "#9aa5a0", "launch": "#3b82c4", "approach": "#2a9d8f", "deter": "#d1495b",
    "return": "#e9a23b", "land": "#7b5ea7", "cooldown": "#b8c0bb",
}


def plot_mission(trace: list, events: list, hits: list, fps: float, target: tuple, title: str, out_png) -> None:
    t = [p[0] for p in trace]
    north = [p[1] for p in trace]
    east = [p[2] for p in trace]
    alt = [p[3] for p in trace]
    state = [p[4] for p in trace]

    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], height_ratios=[3, 0.8, 0.6], hspace=0.35, wspace=0.22)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_alt = fig.add_subplot(grid[0, 1])
    ax_hit = fig.add_subplot(grid[1, 1], sharex=ax_alt)
    ax_txt = fig.add_subplot(grid[2, 1])

    for i in range(1, len(trace)):
        returning = state[i] == "return"
        ax_map.plot(east[i - 1:i + 1], north[i - 1:i + 1], color=STATE_COLORS[state[i]],
                    lw=1.6 if returning else 3.2, ls=(0, (3, 3)) if returning else "-",
                    solid_capstyle="round", zorder=4 if returning else 3)
    ax_map.scatter([0], [0], marker="s", s=110, color="#1c2420", zorder=5)
    ax_map.annotate("dock", (0, 0), textcoords="offset points", xytext=(8, -14), fontsize=10)
    ax_map.scatter([target[1]], [target[0]], marker="*", s=260, color="#d1495b", zorder=5)
    ax_map.annotate("flock", (target[1], target[0]), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(alpha=0.25)
    ax_map.set_title("Top-down path", loc="left", fontsize=11)

    start = 0
    for i in range(1, len(trace) + 1):
        if i == len(trace) or state[i] != state[start]:
            if state[start] != "idle":
                ax_alt.axvspan(t[start], t[min(i, len(t) - 1)], color=STATE_COLORS[state[start]], alpha=0.18, lw=0)
            start = i
    ax_alt.plot(t, alt, color="#1c2420", lw=1.8)
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(alpha=0.25)
    ax_alt.set_title("Altitude and mission phase", loc="left", fontsize=11)
    plt.setp(ax_alt.get_xticklabels(), visible=False)

    hit_times = [i / fps for i, h in enumerate(hits) if h]
    ax_hit.vlines(hit_times, 0, 1, color="#d1495b", lw=0.6)
    ax_hit.set_yticks([])
    ax_hit.set_ylabel("Raven\nnear sheep", fontsize=9, rotation=0, ha="right", va="center")
    ax_hit.set_xlim(0, max(t))
    ax_hit.set_xlabel("Time (s)")

    ax_txt.axis("off")
    key = [e for e in events if e[1] in ("launch", "deter", "return", "land", "cooldown") and e[1] != "cooldown" or "cleared" in e[2] or "gave_up" in e[2]]
    lines = [f"{e[0]:7.1f} s  {e[2]}" for e in key[:5]]
    ax_txt.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace")

    handles = [Line2D([0], [0], color=c, lw=3, label=s) for s, c in STATE_COLORS.items() if s not in ("idle", "cooldown")]
    ax_map.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
