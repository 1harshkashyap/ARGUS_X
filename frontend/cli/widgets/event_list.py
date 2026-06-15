"""Live event feed — DataTable with reordered columns and smart visibility."""
from __future__ import annotations
from datetime import datetime
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Static
from textual.message import Message
from rich.text import Text

# ── Design system colors ──────────────────────────────────────────────
_FG_PRI  = "#e4e4e7"
_FG_SEC  = "#71717a"
_FG_DIM  = "#3f3f46"
_THREAT  = "#ef4444"
_CLEAN   = "#22c55e"
_WARNING = "#f59e0b"
_ACCENT  = "#3b82f6"
_MUTATION = "#a78bfa"

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
    """Event feed with analyst-optimized column order: STATUS → TYPE → TIME → PREVIEW → SCORE → FP."""

    DEFAULT_CSS = """
    EventListWidget {
        width: 28%;
        background: #141417;
        border-right: solid #27272a;
    }
    EventListWidget DataTable {
        background: #141417;
        height: 1fr;
    }
    #ev-header {
        height: 1;
        background: #141417;
        color: #71717a;
        text-style: bold;
        padding: 0 1;
        dock: top;
    }
    #ev-count {
        height: 1;
        background: #09090b;
        color: #71717a;
        padding: 0 1;
        dock: bottom;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: list[dict] = []
        self._selected_row_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("[bold #71717a]EVENT FEED[/]", id="ev-header")
        yield DataTable(id="ev-table", cursor_type="row", show_header=True)
        yield Static("No events yet", id="ev-count")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        # Column order: STATUS → TYPE → TIME → PREVIEW → SCORE → FP
        # Rationale: analyst needs blocked/clean FIRST, threat type SECOND
        table.add_column(Text("ST",      style=_FG_SEC), width=3)
        table.add_column(Text("TY",      style=_FG_SEC), width=4)
        table.add_column(Text("TIME",    style=_FG_SEC), width=9)
        table.add_column(Text("PREVIEW", style=_FG_SEC), width=30)
        table.add_column(Text("SCR",     style=_FG_SEC), width=5)
        table.add_column(Text("FP",      style=_FG_SEC), width=9)

    def load_events(self, events: list[dict]) -> None:
        """Replace event list with fresh data from API."""
        self._events = events
        table = self.query_one(DataTable)
        table.clear()

        for ev in events:
            blocked = ev.get("blocked", False)
            threat  = str(ev.get("threat_type", "CLEAN"))
            abbrev, color = _abbrev(threat)

            # STATUS: single visual mark — ▐ for blocked, dim │ for clean
            if blocked:
                status_text = Text("▐", style=f"bold {_THREAT}")
            else:
                status_text = Text("│", style=_FG_DIM)

            type_text = Text(abbrev, style=f"bold {color}")
            preview   = str(ev.get("message_preview", ""))[:28]
            score     = ev.get("threat_score", 0.0)
            fp_raw    = str(ev.get("attack_fingerprint", ""))[:8]
            time_str  = _parse_time(ev.get("created_at"))

            # Smart visibility: score and FP only shown for blocked events
            if blocked:
                score_color = _THREAT if score > 0.8 else \
                              _WARNING if score > 0.4 else _FG_PRI
                score_text = Text(f"{score:.1f}", style=f"bold {score_color}")
                fp_text    = Text(fp_raw or "—" * 8, style=_FG_SEC)
            else:
                score_text = Text("—", style=_FG_DIM)
                fp_text    = Text("", style=_FG_DIM)

            table.add_row(
                status_text,
                type_text,
                Text(time_str, style=_FG_SEC),
                Text(preview,  style=_FG_PRI),
                score_text,
                fp_text,
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
