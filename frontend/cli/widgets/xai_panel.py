"""XAI Analysis panel — pipeline visual with auto-show last blocked event."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical

# ── Design system colors ──────────────────────────────────────────────
_FG_PRI  = "#e4e4e7"
_FG_SEC  = "#71717a"
_FG_DIM  = "#3f3f46"
_BORDER  = "#27272a"
_ACCENT  = "#3b82f6"
_THREAT  = "#ef4444"
_CLEAN   = "#22c55e"
_WARNING = "#f59e0b"
_MUTATION = "#a78bfa"
_CRITICAL = "#dc2626"

_ACTION_COLORS = {
    "ALLOW":                 _FG_DIM,
    "ALLOW_WITH_MONITORING": _ACCENT,
    "BLOCK":                 _WARNING,
    "BLOCK_AND_MONITOR":     _WARNING,
    "ESCALATE_MONITORING":   _THREAT,
    "TERMINATE_SESSION":     _CRITICAL,
}

_LEVEL_COLORS = {
    "NAIVE":        _FG_SEC,
    "ELEMENTARY":   _ACCENT,
    "INTERMEDIATE": _WARNING,
    "ADVANCED":     _THREAT,
    "APEX":         _MUTATION,
}


def _confidence_bar(value: float, width: int = 16) -> str:
    """Confidence bar with threshold marker at position 14 (~0.87)."""
    filled = round(value * width)
    empty  = width - filled
    color  = _THREAT if value > 0.8 else \
             _WARNING if value > 0.4 else _ACCENT
    # Build bar with threshold marker
    bar_chars = []
    threshold_pos = round(0.87 * width)
    for i in range(width):
        if i < filled:
            bar_chars.append(f"[{color}]█[/]")
        elif i == threshold_pos:
            bar_chars.append(f"[{_FG_SEC}]┊[/]")
        else:
            bar_chars.append(f"[{_BORDER}]░[/]")
    return "".join(bar_chars) + f" [{color}]{value:.0%}[/]"


class XAIPanelWidget(Widget):
    """XAI Analysis — pipeline visual with 3-layer stack connected by box-drawing chars."""

    DEFAULT_CSS = """
    XAIPanelWidget {
        width: 42%;
        background: #141417;
        border-left: solid #3b82f6;
        border-right: solid #27272a;
        padding: 0 1;
        overflow-y: auto;
    }
    #xai-title {
        height: 1;
        color: #71717a;
        text-style: bold;
        dock: top;
    }
    #xai-empty {
        color: #3f3f46;
        text-align: center;
        margin-top: 3;
        text-style: italic;
    }
    #xai-content { height: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Static("[bold #71717a]XAI ANALYSIS[/]", id="xai-title")
        yield Static(
            f"[{_FG_DIM}]Awaiting event selection...\n\n"
            f"Select a blocked event from the feed\n"
            f"to view the full XAI breakdown.[/]",
            id="xai-empty",
        )
        with Vertical(id="xai-content"):
            yield Static("", id="xai-verdict")
            yield Static("", id="xai-soph")
            yield Static("", id="xai-pipeline")
            yield Static("", id="xai-meta")
            yield Static("", id="xai-evolution")

    def on_mount(self) -> None:
        self.query_one("#xai-content").display = False

    def show_event(self, ev: dict) -> None:
        """Update panel to show XAI breakdown for the given event."""
        self.query_one("#xai-empty").display   = False
        self.query_one("#xai-content").display = True

        threat      = str(ev.get("threat_type", "CLEAN"))
        blocked     = ev.get("blocked", False)
        score       = ev.get("threat_score", 0.0)
        soph_score  = ev.get("sophistication_score", 0)
        fp          = ev.get("attack_fingerprint", "")
        explanation = ev.get("explanation") or {}
        layers      = explanation.get("layer_decisions") or []
        primary     = explanation.get("primary_reason", "")
        label       = explanation.get("sophistication_label", "NAIVE")
        action      = explanation.get("recommended_action", "ALLOW")
        evo_note    = explanation.get("evolution_note", "")
        pattern     = explanation.get("pattern_family", "UNKNOWN")

        from widgets.event_list import _THREAT_STYLES
        _, threat_color = _THREAT_STYLES.get(threat, ("??", _FG_SEC))
        label_color = _LEVEL_COLORS.get(label, _FG_SEC)
        action_color = _ACTION_COLORS.get(action, _FG_DIM)

        # ── Verdict headline ──────────────────────────────────────
        if blocked:
            verdict = (
                f"[{_THREAT} bold]█ BLOCKED[/]  "
                f"[{threat_color} bold]{threat.replace('_', ' ')}[/]\n"
                f"[{_FG_PRI} bold]{fp}[/]\n"
                f"[{_FG_SEC}]{primary[:80]}[/]"
            )
        else:
            verdict = (
                f"[{_CLEAN} bold]○ CLEAN[/]  "
                f"[{_FG_SEC}]{threat.replace('_', ' ')}[/]\n"
                f"[{_FG_SEC}]{primary[:80]}[/]"
            )
        self.query_one("#xai-verdict", Static).update(verdict)

        # ── Sophistication indicator ──────────────────────────────
        # 10-position tier indicator: each position = 1 point
        tier_chars = []
        for i in range(10):
            if i < soph_score:
                tier_chars.append(f"[{label_color}]█[/]")
            else:
                tier_chars.append(f"[{_BORDER}]░[/]")
        tier_bar = "".join(tier_chars)

        self.query_one("#xai-soph", Static).update(
            f"\n[{_FG_SEC}]SOPHISTICATION[/]  "
            f"{tier_bar}  "
            f"[{label_color} bold]{soph_score}/10 {label}[/]"
        )

        # ── Layer pipeline (connected with box-drawing) ───────────
        pipeline_lines = []
        for i, layer in enumerate(layers[:3]):
            name    = layer.get("layer_name", f"Layer {i+1}")
            trig    = layer.get("triggered", False)
            conf    = float(layer.get("confidence", 0.0))
            signals = layer.get("signals") or []
            reason  = layer.get("reasoning", "")

            # Pipeline connector
            if i == 0:
                connector = "┌"
            elif i < min(len(layers), 3) - 1:
                connector = "├"
            else:
                connector = "└"

            # Triggered badge
            if trig:
                badge = f"[{_THREAT} bold]TRIGGERED[/]"
            else:
                badge = f"[{_FG_DIM}]CLEAR[/]"

            bar = _confidence_bar(conf)
            sig = signals[0] if signals else ""

            # Sanitize stale ML placeholder messages
            if "Day 7" in sig or "pending" in sig.lower():
                sig = "ML classifier — skipped (firewall resolved)"
            if "Day 7" in reason or "DistilBERT" in reason:
                reason = "ML classifier was not invoked for this request."

            pipeline_lines.append(
                f"\n[{_BORDER}]{connector}── [{_ACCENT}]{name}[/]  {badge}\n"
                f"[{_BORDER}]│[/]   {bar}\n"
                f"[{_BORDER}]│[/]   [{_FG_SEC}]{sig[:55]}[/]"
            )

        self.query_one("#xai-pipeline", Static).update(
            "".join(pipeline_lines)
        )

        # ── Metadata: pattern + action badge ──────────────────────
        action_badge = f"[{action_color} bold] {action} [/]"
        meta = (
            f"\n[{_FG_SEC}]PATTERN[/]  [{threat_color}]{pattern}[/]\n"
            f"[{_FG_SEC}]ACTION [/]  {action_badge}"
        )
        self.query_one("#xai-meta", Static).update(meta)

        # ── Evolution note ────────────────────────────────────────
        if evo_note:
            self.query_one("#xai-evolution", Static).update(
                f"\n[{_MUTATION}]⟳ {evo_note}[/]"
            )
        else:
            self.query_one("#xai-evolution", Static).update("")

    def clear(self) -> None:
        self.query_one("#xai-empty").display   = True
        self.query_one("#xai-content").display = False
