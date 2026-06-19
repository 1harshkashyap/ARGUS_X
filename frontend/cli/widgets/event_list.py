"""Live event feed — DataTable with analyst-optimized columns and smart visibility."""
from __future__ import annotations
from datetime import datetime
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Static
from textual.message import Message
from rich.text import Text

# ── Design system colors (match app.tcss v3.0) ───────────────────────
_FG_PRI    = "#e2e2e8"
_FG_SEC    = "#6b6b7a"
_FG_DIM    = "#3a3a48"
_BORDER    = "#252530"
_THREAT    = "#ff4757"
_CLEAN     = "#2ed573"
_WARNING   = "#ffa502"
_ACCENT    = "#4a9eff"
_MUTATION  = "#a78bfa"

# ── Threat type helpers ───────────────────────────────────────────────

_THREAT_STYLES: dict[str, tuple[str, str]] = {
    "PROMPT_INJECTION":   ("PI", _THREAT),
    "JAILBREAK":          ("JB", _THREAT),
    "DATA_EXFILTRATION":  ("DE", _MUTATION),
    "ROLE_HIJACKING":     ("RH", _WARNING),
    "INDIRECT_INJECTION": ("II", _THREAT),
    "MULTI_TURN":         ("MT", _WARNING),
    "CLEAN":              ("CL", _CLEAN),
}


def _abbrev(threat: str) -> tuple[str, str]:
    return _THREAT_STYLES.get(threat, (threat[:2], _FG_SEC))


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
    """Event feed — columns: ST → TY → SCR → TIME → PREVIEW. FP dropped (shown in XAI only)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: list[dict] = []
        self._selected_row_key: str | None = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="ev-table", cursor_type="row", show_header=True)
        yield Static("No events yet", id="ev-count")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        # Column order: ST → TY → SCR → TIME → PREVIEW
        # Rationale: analyst needs blocked status FIRST, threat type SECOND,
        # severity score THIRD (before time — threat level > chronology),
        # then time, then preview fills remaining width.
        table.add_column(Text("ST",   style=_FG_SEC), width=3)
        table.add_column(Text("TY",   style=_FG_SEC), width=4)
        table.add_column(Text("SCR",  style=_FG_SEC), width=5)
        table.add_column(Text("TIME", style=_FG_SEC), width=9)
        table.add_column(Text("PREVIEW", style=_FG_SEC))

    def load_events(self, events: list[dict]) -> None:
        """Replace event list with fresh data from API."""
        self._events = events
        table = self.query_one(DataTable)
        table.clear()

        for ev in events:
            blocked = ev.get("blocked", False)
            threat  = str(ev.get("threat_type", "CLEAN"))
            abbrev, color = _abbrev(threat)

            # STATUS: single visual mark — ▐ for blocked, dim · for clean
            if blocked:
                status_text = Text("▐", style=f"bold {_THREAT}")
            else:
                status_text = Text("·", style=_FG_DIM)

            type_text = Text(abbrev, style=f"bold {color}")
            preview   = str(ev.get("message_preview", ""))[:35]
            score     = ev.get("threat_score", 0.0)
            time_str  = _parse_time(ev.get("created_at"))

            # Smart visibility: score shown colored for blocked, dim dot for clean
            if blocked:
                score_color = _THREAT if score > 0.8 else \
                              _WARNING if score > 0.4 else _FG_PRI
                score_text = Text(f"{score:.1f}", style=f"bold {score_color}")
            else:
                score_text = Text("·", style=_FG_DIM)

            table.add_row(
                status_text,
                type_text,
                score_text,
                Text(time_str, style=_FG_SEC),
                Text(preview,  style=_FG_PRI if blocked else _FG_SEC),
                key=ev.get("id", preview[:8]),
            )

        count = len(events)
        blocked_count = sum(1 for e in events if e.get("blocked"))
        self.query_one("#ev-count", Static).update(
            f"[{_FG_SEC}]{count} events[/]  "
            f"[{_THREAT}]{blocked_count} blocked[/]"
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
