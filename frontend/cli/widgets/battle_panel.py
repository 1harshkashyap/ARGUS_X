"""Battle Engine panel — live combat stats + controls."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


_TIER_LABELS = {
    1: ("NAIVE",      "#64748b"),
    2: ("EVASIVE",    "#3b82f6"),
    3: ("OBFUSCATED", "#f59e0b"),
    4: ("CONTEXTUAL", "#ef4444"),
    5: ("APEX",       "#a78bfa"),
}

_STRATEGY_COLORS = {
    "NAIVE":      "#64748b",
    "EVASIVE":    "#3b82f6",
    "OBFUSCATED": "#f59e0b",
    "CONTEXTUAL": "#ef4444",
    "APEX":       "#a78bfa",
}


def _tier_bar(tier: int) -> str:
    filled = tier
    empty  = 5 - tier
    _, color = _TIER_LABELS.get(tier, ("?", "#64748b"))
    return (
        f"[{color}]{'█' * filled}[/]"
        f"[#27272a]{'░' * empty}[/]"
    )


class BattlePanelWidget(Widget):
    BORDER_TITLE = "BATTLE ENGINE"
    DEFAULT_CSS = """
    BattlePanelWidget {
        border: solid #27272a;
        background: #18181b;
        width: 30%;
        padding: 1 2;
        overflow-y: auto;
    }
    .bp-label  { color: #64748b; }
    .bp-value  { color: #f2f2f2; text-style: bold; }
    .bp-section-title {
        color: #3b82f6;
        text-style: bold;
        margin-bottom: 1;
    }
    #bp-shortcuts {
        color: #64748b;
        margin-top: 1;
        border-top: solid #27272a;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="bp-tick-tier")
        yield Static("", id="bp-tier-bar")
        yield Static("", id="bp-counters")
        yield Static("", id="bp-last-result")
        yield Static("", id="bp-preview")
        yield Static(
            "\n[#64748b]CONTROLS[/]\n"
            "  [#3b82f6]s[/]  Simulate one cycle\n"
            "  [#3b82f6]p[/]  Pause / Resume\n"
            "  [#3b82f6]F1-F6[/]  Fire preset attack\n"
            "  [#3b82f6]q[/]  Quit",
            id="bp-shortcuts",
        )

    def update_state(self, state: dict) -> None:
        tick     = state.get("tick", 0)
        tier     = max(1, min(5, int(state.get("red_tier", 1))))
        strategy = state.get("red_strategy", "NAIVE")
        attacks  = state.get("red_attacks", 0)
        blocks   = state.get("blue_blocks", 0)
        bypasses = state.get("red_bypasses", 0)
        patches  = state.get("blue_patches", 0)
        last_res = state.get("last_attack_result", "NONE")
        preview  = state.get("current_attack_preview", "")

        _strat_label, strat_color = _TIER_LABELS.get(tier, ("?", "#64748b"))

        self.query_one("#bp-tick-tier", Static).update(
            f"[#64748b]TICK[/]  [bold #f2f2f2]{tick}[/]"
            f"  [#64748b]TIER[/]  [{strat_color}]{tier}/5[/]\n"
            f"[#64748b]STRATEGY[/]  [{strat_color}]{strategy}[/]"
        )

        self.query_one("#bp-tier-bar", Static).update(
            f"\n{_tier_bar(tier)}\n"
        )

        block_rate = f"{blocks / attacks * 100:.0f}%" if attacks else "—"
        self.query_one("#bp-counters", Static).update(
            f"[#64748b]ATTACKS [/]  [bold]{attacks}[/]\n"
            f"[#22c55e]BLOCKS  [/]  [bold #22c55e]{blocks}[/]\n"
            f"[#ef4444]BYPASSES[/]  [bold #ef4444]{bypasses}[/]\n"
            f"[#a78bfa]PATCHES [/]  [bold #a78bfa]{patches}[/]\n"
            f"[#64748b]RATE    [/]  [bold]{block_rate}[/]"
        )

        result_color = "#22c55e" if last_res == "BLOCKED" else \
                       "#ef4444" if last_res == "BYPASSED" else \
                       "#64748b"
        self.query_one("#bp-last-result", Static).update(
            f"\n[#64748b]LAST RESULT[/]  [{result_color}]{last_res}[/]"
        )

        if preview:
            trimmed = preview[:55] + "…" if len(preview) > 55 else preview
            self.query_one("#bp-preview", Static).update(
                f"[#64748b]LAST ATTACK[/]\n"
                f"[#64748b italic]{trimmed}[/]"
            )

    def set_paused(self, paused: bool) -> None:
        """Visual indicator when paused."""
        title = "BATTLE ENGINE  [#f59e0b]PAUSED[/]" if paused \
                else "BATTLE ENGINE"
        self.border_title = title
