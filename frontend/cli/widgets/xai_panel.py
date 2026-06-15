"""XAI Analysis panel — shows full breakdown of selected event."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Label
from textual.containers import Vertical


_ACTION_COLORS = {
    "ALLOW":                 "#22c55e",
    "ALLOW_WITH_MONITORING": "#22c55e",
    "BLOCK":                 "#ef4444",
    "BLOCK_AND_MONITOR":     "#f59e0b",
    "ESCALATE_MONITORING":   "#f59e0b",
    "TERMINATE_SESSION":     "#ef4444",
}

_LEVEL_COLORS = {
    "NAIVE":        "#64748b",
    "ELEMENTARY":   "#3b82f6",
    "INTERMEDIATE": "#f59e0b",
    "ADVANCED":     "#ef4444",
    "APEX":         "#a78bfa",
}


def _confidence_bar(value: float, width: int = 14) -> str:
    filled = round(value * width)
    empty  = width - filled
    color  = "#ef4444" if value > 0.8 else \
             "#f59e0b" if value > 0.4 else \
             "#3b82f6"
    return f"[{color}]{'█' * filled}[/][#27272a]{'░' * empty}[/] [{color}]{value:.0%}[/]"


class XAIPanelWidget(Widget):
    BORDER_TITLE = "XAI ANALYSIS"
    DEFAULT_CSS = """
    XAIPanelWidget {
        border: solid #27272a;
        background: #18181b;
        width: 38%;
        padding: 1 2;
        overflow-y: auto;
    }
    #xai-empty {
        color: #64748b;
        text-align: center;
        margin-top: 4;
    }
    #xai-content { height: auto; }
    .xai-section-title {
        color: #64748b;
        text-style: bold;
        margin-top: 1;
    }
    .xai-row { margin-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Select an event from the list\n\n"
            "← Use arrow keys or click to select",
            id="xai-empty",
        )
        with Vertical(id="xai-content"):
            yield Static("", id="xai-header")
            yield Static("", id="xai-score-bar")
            yield Static("", id="xai-layer1")
            yield Static("", id="xai-layer2")
            yield Static("", id="xai-layer3")
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
        _, threat_color = _THREAT_STYLES.get(threat, ("??", "#64748b"))
        label_color = _LEVEL_COLORS.get(label, "#64748b")
        action_color = _ACTION_COLORS.get(action, "#64748b")

        # Header
        status_str = "■ BLOCKED" if blocked else "○ CLEAN"
        status_color = "#ef4444" if blocked else "#22c55e"
        self.query_one("#xai-header", Static).update(
            f"[{status_color}]{status_str}[/]  "
            f"[{threat_color}]{threat.replace('_', ' ')}[/]\n"
            f"[#64748b]Primary:[/] {primary}"
        )

        # Sophistication bar
        # 20-char bar gives better visual weight even at score=1
        soph_filled = soph_score * 2
        soph_empty  = (10 - soph_score) * 2
        soph_bar = (
            f"[{label_color}]{'█' * soph_filled}[/]"
            f"[#27272a]{'░' * soph_empty}[/]"
        )
        self.query_one("#xai-score-bar", Static).update(
            f"\n[#64748b]SOPHISTICATION[/]  "
            f"{soph_bar}  "
            f"[{label_color}]{soph_score}/10  {label}[/]"
        )

        # Layer decisions
        layer_ids = ["#xai-layer1", "#xai-layer2", "#xai-layer3"]
        for i, lid in enumerate(layer_ids):
            widget = self.query_one(lid, Static)
            if i < len(layers):
                layer   = layers[i]
                name    = layer.get("layer_name", f"Layer {i+1}")
                trig    = layer.get("triggered", False)
                conf    = float(layer.get("confidence", 0.0))
                signals = layer.get("signals") or []
                reason  = layer.get("reasoning", "")

                trig_str = (
                    f"[[bold #ef4444]TRIGGERED[/]]" if trig
                    else "[[#64748b]CLEAR[/]]    "
                )
                bar = _confidence_bar(conf)
                sig = signals[0] if signals else ""

                # Update outdated ML placeholder messages
                if "Day 7" in sig or "pending" in sig.lower():
                    sig = "ML disabled — set ML_ENABLED=true to activate"
                if "Day 7" in reason or "DistilBERT" in reason:
                    reason = "ML semantic classifier is currently disabled (ML_ENABLED=false)."

                widget.update(
                    f"\n[#3b82f6]{name}[/]  {trig_str}\n"
                    f"  {bar}\n"
                    f"  [#64748b]{sig[:60]}[/]"
                )
            else:
                widget.update("")

        # Metadata
        meta_lines = []
        if fp:
            meta_lines.append(
                f"[#64748b]FINGERPRINT[/]  [bold #e2e8f0]{fp}[/]"
            )
        meta_lines.append(
            f"[#64748b]PATTERN    [/]  [{threat_color}]{pattern}[/]"
        )
        meta_lines.append(
            f"[#64748b]ACTION     [/]  [{action_color}]{action}[/]"
        )
        self.query_one("#xai-meta", Static).update(
            "\n" + "\n".join(meta_lines)
        )

        # Evolution note
        if evo_note:
            self.query_one("#xai-evolution", Static).update(
                f"\n[#a78bfa]⟳ {evo_note}[/]"
            )
        else:
            self.query_one("#xai-evolution", Static).update("")

    def clear(self) -> None:
        self.query_one("#xai-empty").display   = True
        self.query_one("#xai-content").display = False
