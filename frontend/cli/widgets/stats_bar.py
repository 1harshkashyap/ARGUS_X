"""Stats bar — single-line compact metrics bar with rate sparkline."""
from __future__ import annotations
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static
from theme import (
    VOID, SURFACE, ACCENT, FG_PRIMARY, FG_SECONDARY, FG_DIM, BORDER,
    THREAT, STATUS_CLEAN, STATUS_WARNING,
)


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
    """Single-line stats bar with rate trend sparkline."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._online: bool = False
        self._rate_history: list[float] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-line")

    def on_mount(self) -> None:
        self.query_one("#sb-line", Static).update(
            f"[{FG_DIM}]◌[/] [{ACCENT} bold]ARGUS-X[/]  "
            f"[{FG_SECONDARY}]Connecting...[/]"
        )

    def set_online(self, online: bool) -> None:
        self._online = online
        if not online:
            self.query_one("#sb-line", Static).update(
                f"[{THREAT}]✕[/] [{ACCENT} bold]ARGUS-X[/]  "
                f"[{THREAT}]OFFLINE[/]"
            )

    def _sparkline(self, values: list[float]) -> str:
        """Render up to 8 values as a Unicode block sparkline."""
        if len(values) < 2:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        chars = []
        for v in values[-8:]:
            idx = int((v - lo) / span * (len(blocks) - 1))
            chars.append(blocks[idx])
        return "".join(chars)

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

        # Health dot
        dot = f"[{STATUS_CLEAN}]●[/]" if self._online else f"[{THREAT}]✕[/]"

        # Rate color — escalates with threat level
        rate_pct = f"{rate * 100:.0f}%"
        if rate > 0.5:
            rate_c = THREAT
        elif rate > 0.2:
            rate_c = STATUS_WARNING
        else:
            rate_c = STATUS_CLEAN

        # Rate sparkline trend
        self._rate_history.append(rate)
        if len(self._rate_history) > 8:
            self._rate_history = self._rate_history[-8:]
        trend = self._sparkline(self._rate_history)

        # Service dots
        ml_dot = f"[{STATUS_CLEAN}]●[/]" if ml_svc == "online" else \
                 f"[{STATUS_WARNING}]◉[/]" if ml_svc == "degraded" else \
                 f"[{FG_DIM}]○[/]"
        bt_dot = f"[{STATUS_CLEAN}]●[/]" if bt_svc not in ("disabled", "—") else \
                 f"[{FG_DIM}]○[/]"

        # Bypass highlight
        byp_c = THREAT if bypasses > 0 else FG_SECONDARY

        line = (
            f"{dot} [{ACCENT} bold]ARGUS-X[/]  "
            f"[{BORDER}]│[/] "
            f"[{THREAT} bold]{blocked}[/][{FG_SECONDARY}] BLK[/]  "
            f"[{FG_PRIMARY}]{total}[/][{FG_SECONDARY}] evt[/]  "
            f"[{STATUS_CLEAN}]{clean}[/][{FG_SECONDARY}] cln[/]  "
            f"[{byp_c}]{bypasses}[/][{FG_SECONDARY}] byp[/]  "
            f"[{BORDER}]│[/] "
            f"[{rate_c} bold]{rate_pct}[/] "
            f"[{FG_SECONDARY}]{trend}[/]  "
            f"[{FG_PRIMARY}]{_fmt_uptime(uptime)}[/][{FG_SECONDARY}] up[/]  "
            f"[{BORDER}]│[/] "
            f"{ml_dot}[{FG_SECONDARY}]ml[/] "
            f"{bt_dot}[{FG_SECONDARY}]btl[/]"
        )

        self.query_one("#sb-line", Static).update(line)
