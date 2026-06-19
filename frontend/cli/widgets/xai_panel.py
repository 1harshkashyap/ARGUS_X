"""XAI Analysis panel — pipeline visual with auto-show last blocked event."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical
from theme import (
    FG_PRIMARY, FG_SECONDARY, FG_DIM, BORDER, ACCENT,
    THREAT, STATUS_CLEAN, STATUS_WARNING, STATUS_MUTATION, STATUS_CRITICAL,
    THREAT_TYPE_STYLES, ACTION_COLORS, LEVEL_COLORS,
)


def _confidence_bar(value: float, width: int = 16) -> str:
    """Confidence bar with threshold marker at FIREWALL_ML_THRESHOLD (0.87)."""
    filled = round(value * width)
    color  = THREAT if value > 0.8 else \
             STATUS_WARNING if value > 0.4 else ACCENT
    threshold_pos = round(0.87 * width)
    bar_chars = []
    for i in range(width):
        if i < filled:
            bar_chars.append(f"[{color}]█[/]")
        elif i == threshold_pos:
            bar_chars.append(f"[{FG_SECONDARY}]┊[/]")
        else:
            bar_chars.append(f"[{BORDER}]░[/]")
    return "".join(bar_chars) + f" [{color}]{value:.0%}[/]"


class XAIPanelWidget(Widget):
    """XAI Analysis — pipeline visual with 3-layer stack."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._showing_event: bool = False

    def is_empty(self) -> bool:
        """Returns True if no event is currently displayed."""
        return not self._showing_event

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {FG_SECONDARY}]XAI ANALYSIS[/]", id="xai-title")
        yield Static(
            f"[{FG_DIM}]Awaiting event selection...\n\n"
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
        self._showing_event = True
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

        _, threat_color = THREAT_TYPE_STYLES.get(threat, ("??", FG_SECONDARY))
        label_color = LEVEL_COLORS.get(label, FG_SECONDARY)
        action_color = ACTION_COLORS.get(action, FG_DIM)

        # ── Verdict headline (T1 — most visually dominant) ────────
        if blocked:
            verdict = (
                f"[{THREAT} bold]█ BLOCKED[/]  "
                f"[{threat_color} bold]{threat.replace('_', ' ')}[/]\n"
                f"[{FG_PRIMARY} bold]{fp}[/]\n"
                f"[{FG_SECONDARY}]{primary[:80]}[/]"
            )
        else:
            verdict = (
                f"[{STATUS_CLEAN} bold]○ CLEAN[/]  "
                f"[{FG_SECONDARY}]{threat.replace('_', ' ')}[/]\n"
                f"[{FG_SECONDARY}]{primary[:80]}[/]"
            )
        self.query_one("#xai-verdict", Static).update(verdict)

        # ── Sophistication indicator (10-position tier bar) ───────
        tier_chars = []
        for i in range(10):
            if i < soph_score:
                tier_chars.append(f"[{label_color}]█[/]")
            else:
                tier_chars.append(f"[{BORDER}]░[/]")
        tier_bar = "".join(tier_chars)

        self.query_one("#xai-soph", Static).update(
            f"\n[{FG_SECONDARY}]SOPHISTICATION[/]  "
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

            if i == 0:
                connector = "┌"
            elif i < min(len(layers), 3) - 1:
                connector = "├"
            else:
                connector = "└"

            if trig:
                badge = f"[{THREAT} bold]TRIGGERED[/]"
            else:
                badge = f"[{FG_DIM}]CLEAR[/]"

            bar = _confidence_bar(conf)
            sig = signals[0] if signals else ""

            # Sanitize stale ML placeholder messages
            if "Day 7" in sig or "pending" in sig.lower():
                sig = "ML classifier — skipped (firewall resolved)"
            if "Day 7" in reason or "DistilBERT" in reason:
                reason = "ML classifier was not invoked for this request."

            pipeline_lines.append(
                f"\n[{BORDER}]{connector}── [{ACCENT}]{name}[/]  {badge}\n"
                f"[{BORDER}]│[/]   {bar}\n"
                f"[{BORDER}]│[/]   [{FG_SECONDARY}]{sig[:55]}[/]"
            )

        self.query_one("#xai-pipeline", Static).update(
            "".join(pipeline_lines)
        )

        # ── Metadata: pattern + action badge ──────────────────────
        action_badge = f"[{action_color} bold] {action} [/]"
        meta = (
            f"\n[{FG_SECONDARY}]PATTERN[/]  [{threat_color}]{pattern}[/]\n"
            f"[{FG_SECONDARY}]ACTION [/]  {action_badge}"
        )
        self.query_one("#xai-meta", Static).update(meta)

        # ── Evolution note ────────────────────────────────────────
        if evo_note:
            self.query_one("#xai-evolution", Static).update(
                f"\n[{STATUS_MUTATION}]⟳ {evo_note}[/]"
            )
        else:
            self.query_one("#xai-evolution", Static).update("")

    def clear(self) -> None:
        self._showing_event = False
        self.query_one("#xai-empty").display   = True
        self.query_one("#xai-content").display = False
