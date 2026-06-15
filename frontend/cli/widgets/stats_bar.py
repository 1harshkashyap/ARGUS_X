"""Stats bar — docked at top, shows live system metrics."""
from __future__ import annotations
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_rate(rate: float) -> str:
    return f"{rate * 100:.1f}%"


class StatPill(Static):
    DEFAULT_CSS = """
    StatPill {
        padding: 0 2;
        border: solid $panel-lighten-1;
        height: 1;
        margin-right: 1;
    }
    """


class StatsBar(Widget):
    DEFAULT_CSS = """
    StatsBar {
        height: 3;
        background: #18181b;
        border-bottom: solid #27272a;
        layout: horizontal;
        align: center middle;
        padding: 0 2;
    }
    #sb-logo {
        color: #3b82f6;
        text-style: bold;
        width: 12;
        padding-right: 2;
    }
    #sb-health-dot {
        width: 3;
        padding-right: 1;
    }
    .sb-label { color: #64748b; }
    .sb-value { color: #f2f2f2; text-style: bold; }
    .sb-value-bad  { color: #ef4444; text-style: bold; }
    .sb-value-good { color: #22c55e; text-style: bold; }
    .sb-sep { color: #27272a; width: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Static("ARGUS-X", id="sb-logo")
        yield Static("●", id="sb-health-dot")
        yield Static("[#64748b]EVENTS[/] [bold]—[/]",    id="sb-events")
        yield Static(" [#27272a]|[/] ", classes="sb-sep")
        yield Static("[#64748b]BLOCKED[/] [bold]—[/]",   id="sb-blocked")
        yield Static(" [#27272a]|[/] ", classes="sb-sep")
        yield Static("[#64748b]CLEAN[/] [bold]—[/]",     id="sb-clean")
        yield Static(" [#27272a]|[/] ", classes="sb-sep")
        yield Static("[#64748b]RATE[/] [bold]—[/]",      id="sb-rate")
        yield Static(" [#27272a]|[/] ", classes="sb-sep")
        yield Static("[#64748b]UPTIME[/] [bold]—[/]",    id="sb-uptime")
        yield Static(" [#27272a]|[/] ", classes="sb-sep")
        yield Static("[#64748b]ML[/] [bold]—[/]",        id="sb-ml")
        yield Static(" [#27272a]|[/] ", classes="sb-sep")
        yield Static("[#64748b]BATTLE[/] [bold]—[/]",    id="sb-battle")

    def set_online(self, online: bool) -> None:
        dot = self.query_one("#sb-health-dot", Static)
        if online:
            dot.update("[#22c55e]●[/]")
        else:
            dot.update("[#ef4444]●[/]")

    def update_stats(self, data: dict) -> None:
        total   = data.get("total_events", 0)
        blocked = data.get("total_blocked", 0)
        clean   = data.get("total_clean", 0)
        rate    = data.get("block_rate", 0.0)
        uptime  = data.get("uptime_seconds", 0.0)
        ml_svc  = data.get("services", {}).get("ml_classifier", "—")
        bt_svc  = data.get("services", {}).get("battle_engine", "—")

        self.query_one("#sb-events",  Static).update(
            f"[#64748b]EVENTS[/] [bold]{total}[/]"
        )
        blocked_color = "#ef4444" if blocked > 0 else "#f2f2f2"
        self.query_one("#sb-blocked", Static).update(
            f"[#64748b]BLOCKED[/] [{blocked_color}]{blocked}[/]"
        )
        self.query_one("#sb-clean",   Static).update(
            f"[#64748b]CLEAN[/] [#22c55e]{clean}[/]"
        )
        rate_color = "#ef4444" if rate > 0.5 else "#f59e0b" if rate > 0.2 else "#22c55e"
        self.query_one("#sb-rate",    Static).update(
            f"[#64748b]RATE[/] [{rate_color}]{_fmt_rate(rate)}[/]"
        )
        self.query_one("#sb-uptime",  Static).update(
            f"[#64748b]UPTIME[/] [bold]{_fmt_uptime(uptime)}[/]"
        )

        ml_color = "#22c55e" if ml_svc == "online" else \
                   "#f59e0b" if ml_svc == "degraded" else "#64748b"
        self.query_one("#sb-ml",      Static).update(
            f"[#64748b]ML[/] [{ml_color}]{ml_svc.upper()}[/]"
        )
        bt_color = "#22c55e" if bt_svc != "disabled" else "#64748b"
        self.query_one("#sb-battle",  Static).update(
            f"[#64748b]BATTLE[/] [{bt_color}]{bt_svc.upper()}[/]"
        )
