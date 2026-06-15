"""Stats bar — single Rich markup string, always fits on one line."""
from __future__ import annotations
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h{m}m"
    if m: return f"{m}m{s}s"
    return f"{s}s"


def _fmt_rate(rate: float) -> str:
    return f"{rate * 100:.1f}%"


class StatsBar(Widget):
    DEFAULT_CSS = """
    StatsBar {
        height: 3;
        background: #18181b;
        border-bottom: solid #27272a;
        padding: 0 1;
        layout: vertical;
        align: left middle;
    }
    #sb-line1 { height: 1; }
    #sb-line2 { height: 1; color: #64748b; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-line1")
        yield Static("", id="sb-line2")

    def on_mount(self) -> None:
        self.query_one("#sb-line1", Static).update(
            "[bold #3b82f6]ARGUS-X[/]  "
            "[#64748b]● Connecting...[/]"
        )

    def set_online(self, online: bool) -> None:
        # Updated via update_stats — dot color set there
        pass

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
        dot_color = "#22c55e"

        # Rate color
        rate_color = "#ef4444" if rate > 0.5 else \
                     "#f59e0b" if rate > 0.2 else "#22c55e"

        # ML color
        ml_color  = "#22c55e"  if ml_svc  == "online"   else \
                    "#f59e0b"  if ml_svc  == "degraded" else "#64748b"

        # Battle color
        bt_color  = "#22c55e" if bt_svc not in ("disabled", "—") else "#64748b"

        line1 = (
            f"[bold #3b82f6]ARGUS-X[/]  "
            f"[{dot_color}]●[/]  "
            f"[#64748b]EVENTS[/] [bold #f2f2f2]{total}[/]  "
            f"[#27272a]│[/]  "
            f"[#64748b]BLOCKED[/] [bold #ef4444]{blocked}[/]  "
            f"[#27272a]│[/]  "
            f"[#64748b]CLEAN[/] [bold #22c55e]{clean}[/]  "
            f"[#27272a]│[/]  "
            f"[#64748b]BYPASSED[/] [bold]{bypasses}[/]  "
            f"[#27272a]│[/]  "
            f"[#64748b]RATE[/] [{rate_color}]{_fmt_rate(rate)}[/]  "
            f"[#27272a]│[/]  "
            f"[#64748b]UP[/] [bold]{_fmt_uptime(uptime)}[/]"
        )
        line2 = (
            f"[#64748b]ML[/] [{ml_color}]{ml_svc.upper()}[/]  "
            f"[#27272a]│[/]  "
            f"[#64748b]BATTLE[/] [{bt_color}]{bt_svc.upper()}[/]"
        )

        self.query_one("#sb-line1", Static).update(line1)
        self.query_one("#sb-line2", Static).update(line2)
