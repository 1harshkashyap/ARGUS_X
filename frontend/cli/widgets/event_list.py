"""Live event list — DataTable with real-time polling."""
from __future__ import annotations
from datetime import datetime
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Label
from textual.message import Message
from rich.text import Text


# ── Threat type helpers ───────────────────────────────────────────────

_THREAT_STYLES: dict[str, tuple[str, str]] = {
    "PROMPT_INJECTION":   ("PI", "#ef4444"),
    "JAILBREAK":          ("JB", "#ef4444"),
    "DATA_EXFILTRATION":  ("DE", "#a78bfa"),
    "ROLE_HIJACKING":     ("RH", "#f59e0b"),
    "INDIRECT_INJECTION": ("II", "#ef4444"),
    "MULTI_TURN":         ("MT", "#f59e0b"),
    "CLEAN":              ("CL", "#22c55e"),
}


def _abbrev(threat: str) -> tuple[str, str]:
    return _THREAT_STYLES.get(threat, (threat[:2], "#64748b"))


def _parse_time(iso: str | None) -> str:
    if not iso:
        return "--:--:--"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


class EventSelected(Message):
    """Posted when user selects a row. Carries the raw event dict."""
    def __init__(self, event: dict) -> None:
        self.event = event
        super().__init__()


class EventListWidget(Widget):
    BORDER_TITLE = "LIVE EVENTS"
    DEFAULT_CSS = """
    EventListWidget {
        border: solid #27272a;
        background: #18181b;
        width: 32%;
    }
    EventListWidget DataTable {
        background: #18181b;
        height: 1fr;
    }
    #ev-count {
        height: 1;
        background: #0e0e10;
        color: #64748b;
        padding: 0 1;
        dock: bottom;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: list[dict] = []
        self._selected_row_key: str | None = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="ev-table", cursor_type="row", show_header=True)
        yield Label("No events yet", id="ev-count")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns(
            Text("TIME",    style="#64748b"),
            Text("STATUS",  style="#64748b"),
            Text("TYPE",    style="#64748b"),
            Text("PREVIEW", style="#64748b"),
            Text("SCORE",   style="#64748b"),
            Text("ID",      style="#64748b"),
        )

    def load_events(self, events: list[dict]) -> None:
        """Replace event list with fresh data from API."""
        self._events = events
        table = self.query_one(DataTable)
        table.clear()

        for ev in events:
            blocked = ev.get("blocked", False)
            threat  = str(ev.get("threat_type", "CLEAN"))
            abbrev, color = _abbrev(threat)

            status_text = Text("● BLOCKED", style="bold #ef4444") if blocked \
                     else Text("○ CLEAN",   style="bold #22c55e")
            type_text   = Text(abbrev, style=f"bold {color}")
            preview     = str(ev.get("message_preview", ""))[:38]
            score       = ev.get("threat_score", 0.0)
            score_color = "#ef4444" if score > 0.8 else \
                          "#f59e0b" if score > 0.4 else "#22c55e"
            fp          = str(ev.get("attack_fingerprint", ""))[:8] or "—" * 8
            time_str    = _parse_time(ev.get("created_at"))

            table.add_row(
                Text(time_str,       style="#64748b"),
                status_text,
                type_text,
                Text(preview,        style="#e2e8f0"),
                Text(f"{score:.2f}", style=score_color),
                Text(fp,             style="#64748b"),
                key=ev.get("id", preview[:8]),
            )

        count = len(events)
        blocked_count = sum(1 for e in events if e.get("blocked"))
        self.query_one("#ev-count", Label).update(
            f"[#64748b]{count} events · {blocked_count} blocked[/]"
        )

        if table.row_count > 0:
            table.move_cursor(row=0)

    def prepend_event(self, ev: dict) -> None:
        """Add a single new event to the top without full reload."""
        self._events.insert(0, ev)
        if len(self._events) > 100:
            self._events = self._events[:100]
        self.load_events(self._events)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_idx = event.cursor_row
        if 0 <= row_idx < len(self._events):
            self.post_message(EventSelected(self._events[row_idx]))
