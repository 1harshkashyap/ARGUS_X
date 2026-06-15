"""Battle Engine panel — Red vs Blue contest view. No static controls."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

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

_TIER_LABELS = {
    1: ("NAIVE",      _FG_SEC),
    2: ("EVASIVE",    _ACCENT),
    3: ("OBFUSCATED", _WARNING),
    4: ("CONTEXTUAL", _THREAT),
    5: ("APEX",       _MUTATION),
}


def _tier_bar(tier: int) -> str:
    """10-char tier bar — each tier = 2 blocks."""
    _, color = _TIER_LABELS.get(tier, ("?", _FG_SEC))
    filled = tier * 2
    empty  = (5 - tier) * 2
    return (
        f"[{color}]{'█' * filled}[/]"
        f"[{_BORDER}]{'░' * empty}[/]"
    )


class BattlePanelWidget(Widget):
    """Battle engine — Red vs Blue metrics, bypass-first hierarchy."""

    DEFAULT_CSS = """
    BattlePanelWidget {
        width: 30%;
        background: #141417;
        padding: 0 1;
        overflow-y: auto;
    }
    #bp-title {
        height: 1;
        color: #71717a;
        text-style: bold;
        dock: top;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("[bold #71717a]BATTLE ENGINE[/]", id="bp-title")
        yield Static("", id="bp-headline")
        yield Static("", id="bp-tier-bar")
        yield Static("", id="bp-contest")
        yield Static("", id="bp-last-result")
        yield Static("", id="bp-preview")

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

        strat_label, strat_color = _TIER_LABELS.get(tier, ("?", _FG_SEC))

        # Block rate is the headline number
        block_rate = f"{blocks / attacks * 100:.0f}%" if attacks else "—"
        rate_color = _CLEAN if attacks and blocks / attacks > 0.9 else \
                     _WARNING if attacks and blocks / attacks > 0.5 else \
                     _THREAT if attacks else _FG_SEC

        # Bypass count is T1 THREAT when >0
        byp_color = _THREAT if bypasses > 0 else _FG_SEC

        self.query_one("#bp-headline", Static).update(
            f"\n[{_FG_SEC}]BLOCK RATE[/]  "
            f"[{rate_color} bold]{block_rate}[/]  "
            f"[{_FG_SEC}]tick[/] [{_FG_PRI}]{tick}[/]  "
            f"[{_FG_SEC}]tier[/] [{strat_color}]{tier}/5 {strategy}[/]"
        )

        self.query_one("#bp-tier-bar", Static).update(
            f"{_tier_bar(tier)}  [{strat_color}]{strat_label}[/]"
        )

        # Red vs Blue contest — two columns
        self.query_one("#bp-contest", Static).update(
            f"\n[{_THREAT} bold]RED[/]                [{_CLEAN} bold]BLUE[/]\n"
            f"[{_FG_SEC}]attacks[/]  [{_FG_PRI} bold]{attacks:<8}[/]"
            f"[{_FG_SEC}]blocks [/]  [{_CLEAN} bold]{blocks}[/]\n"
            f"[{_FG_SEC}]bypass [/]  [{byp_color} bold]{bypasses:<8}[/]"
            f"[{_FG_SEC}]patches[/]  [{_MUTATION} bold]{patches}[/]"
        )

        result_color = _CLEAN if last_res == "BLOCKED" else \
                       _THREAT if last_res == "BYPASSED" else _FG_DIM
        self.query_one("#bp-last-result", Static).update(
            f"\n[{_FG_SEC}]LAST[/]  [{result_color} bold]{last_res}[/]"
        )

        if preview:
            trimmed = preview[:60] + "…" if len(preview) > 60 else preview
            self.query_one("#bp-preview", Static).update(
                f"\n[{_FG_SEC}]LAST ATTACK[/]\n"
                f"[{_FG_DIM} italic]{trimmed}[/]"
            )

    def set_paused(self, paused: bool) -> None:
        """Visual indicator when paused."""
        if paused:
            self.query_one("#bp-title", Static).update(
                f"[bold #71717a]BATTLE ENGINE[/]  [{_WARNING} bold]▌▌ PAUSED[/]"
            )
        else:
            self.query_one("#bp-title", Static).update(
                "[bold #71717a]BATTLE ENGINE[/]"
            )
