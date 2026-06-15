"""Stats bar — single-line compact metrics bar. SOC command center style."""
from __future__ import annotations
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

# ── Design system colors (match app.tcss) ─────────────────────────────
_VOID      = "#09090b"
_SURFACE   = "#141417"
_ACCENT    = "#3b82f6"
_FG_PRI    = "#e4e4e7"
_FG_SEC    = "#71717a"
_FG_DIM    = "#3f3f46"
_BORDER    = "#27272a"
_THREAT    = "#ef4444"
_CLEAN     = "#22c55e"
_WARNING   = "#f59e0b"


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m}m"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


class StatsBar(Widget):
    """Single-line stats bar. Health dot → logo → blocked → events → clean → bypass → rate → uptime → service dots."""

    DEFAULT_CSS = """
    StatsBar {
        height: 2;
        background: #141417;
        border-bottom: solid #27272a;
        padding: 0 1;
        layout: vertical;
        content-align: left middle;
    }
    #sb-line { height: 1; width: 1fr; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._online: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-line")

    def on_mount(self) -> None:
        self.query_one("#sb-line", Static).update(
            f"[{_FG_DIM}]●[/] [{_ACCENT} bold]ARGUS-X[/]  "
            f"[{_FG_SEC}]Connecting...[/]"
        )

    def set_online(self, online: bool) -> None:
        self._online = online
        if not online:
            self.query_one("#sb-line", Static).update(
                f"[{_THREAT}]✕[/] [{_ACCENT} bold]ARGUS-X[/]  "
                f"[{_THREAT}]OFFLINE[/]"
            )

    def update_stats(self, data: dict) -> None:
        total    = data.get("total_events",  0)
        blocked  = data.get("total_blocked", 0)
        clean    = data.get("total_clean",   0)
        bypasses = data.get("total_bypasses", 0)
        rate     = data.get("block_rate",    0.0)
        uptime   = data.get("uptime_seconds", 0.0)
        services = data.get("services",      {})
        ml_svc   = services.get("ml_classifier", "—")
        bt_svc   = services.get("battle_engine",  "—")

        # Health dot — reflects actual connection status
        dot = f"[{_CLEAN}]●[/]" if self._online else f"[{_THREAT}]✕[/]"

        # Rate color — escalates with threat level
        rate_pct = f"{rate * 100:.0f}%"
        if rate > 0.5:
            rate_c = _THREAT
        elif rate > 0.2:
            rate_c = _WARNING
        else:
            rate_c = _CLEAN

        # Service dots — rightmost for glanceable status
        ml_dot = f"[{_CLEAN}]●[/]" if ml_svc == "online" else \
                 f"[{_WARNING}]◉[/]" if ml_svc == "degraded" else \
                 f"[{_FG_DIM}]○[/]"
        bt_dot = f"[{_CLEAN}]●[/]" if bt_svc not in ("disabled", "—") else \
                 f"[{_FG_DIM}]○[/]"

        # Bypass highlight — if bypasses > 0, show in threat color
        byp_c = _THREAT if bypasses > 0 else _FG_SEC

        line = (
            f"{dot} [{_ACCENT} bold]ARGUS-X[/]  "
            f"[{_BORDER}]│[/] "
            f"[{_THREAT} bold]{blocked}[/][{_FG_SEC}] blk[/]  "
            f"[{_FG_PRI} bold]{total}[/][{_FG_SEC}] evt[/]  "
            f"[{_CLEAN}]{clean}[/][{_FG_SEC}] cln[/]  "
            f"[{byp_c}]{bypasses}[/][{_FG_SEC}] byp[/]  "
            f"[{_BORDER}]│[/] "
            f"[{rate_c} bold]{rate_pct}[/][{_FG_SEC}] rate[/]  "
            f"[{_FG_PRI}]{_fmt_uptime(uptime)}[/][{_FG_SEC}] up[/]  "
            f"[{_BORDER}]│[/] "
            f"{ml_dot}[{_FG_SEC}]ml[/] "
            f"{bt_dot}[{_FG_SEC}]btl[/]"
        )

        self.query_one("#sb-line", Static).update(line)
