"""Live event feed — DataTable with analyst-optimized columns and smart visibility."""
from __future__ import annotations
from datetime import datetime
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Static
from textual.message import Message
from rich.text import Text
from theme import (
    FG_PRIMARY, FG_SECONDARY, FG_DIM, BORDER, ACCENT,
    THREAT, STATUS_CLEAN, STATUS_WARNING, STATUS_MUTATION,
    THREAT_TYPE_STYLES,
)


def _abbrev(threat: str) -> tuple[str, str]:
    return THREAT_TYPE_STYLES.get(threat, (threat[:2], FG_SECONDARY))


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
    """Event feed — columns: ST → TY → SCR → TIME → PREVIEW."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: list[dict] = []
        self._selected_row_key: str | None = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="ev-table", cursor_type="row", show_header=True)
        yield Static("No events yet", id="ev-count")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column(Text("ST",      style=FG_SECONDARY), width=3)
        table.add_column(Text("TY",      style=FG_SECONDARY), width=4)
        table.add_column(Text("SCR",     style=FG_SECONDARY), width=5)
        table.add_column(Text("TIME",    style=FG_SECONDARY), width=9)
        table.add_column(Text("PREVIEW", style=FG_SECONDARY))

    def load_events(self, events: list[dict]) -> None:
        """Replace event list with fresh data from API."""
        self._events = events
        table = self.query_one(DataTable)
        table.clear()

        for ev in events:
            blocked = ev.get("blocked", False)
            threat  = str(ev.get("threat_type", "CLEAN"))
            abbrev, color = _abbrev(threat)

            if blocked:
                status_text = Text("▐", style=f"bold {THREAT}")
            else:
                status_text = Text("·", style=FG_DIM)

            type_text = Text(abbrev, style=f"bold {color}")
            preview   = str(ev.get("message_preview", ""))[:35]
            score     = ev.get("threat_score", 0.0)
            time_str  = _parse_time(ev.get("created_at"))

            if blocked:
                score_color = THREAT if score > 0.8 else \
                              STATUS_WARNING if score > 0.4 else FG_PRIMARY
                score_text = Text(f"{score:.1f}", style=f"bold {score_color}")
            else:
                score_text = Text("·", style=FG_DIM)

            table.add_row(
                status_text,
                type_text,
                score_text,
                Text(time_str, style=FG_SECONDARY),
                Text(preview,  style=FG_PRIMARY if blocked else FG_SECONDARY),
                key=ev.get("id", preview[:8]),
            )

        count = len(events)
        blocked_count = sum(1 for e in events if e.get("blocked"))
        self.query_one("#ev-count", Static).update(
            f"[{FG_SECONDARY}]{count} events[/]  "
            f"[{THREAT}]{blocked_count} blocked[/]"
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
