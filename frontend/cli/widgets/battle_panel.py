"""Battle Engine panel — Red vs Blue contest view. Bypass-first hierarchy."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from theme import (
    FG_PRIMARY, FG_SECONDARY, FG_DIM, BORDER, ACCENT,
    THREAT, STATUS_CLEAN, STATUS_WARNING, STATUS_MUTATION,
)

_TIER_LABELS = {
    1: ("NAIVE",      FG_SECONDARY),
    2: ("EVASIVE",    ACCENT),
    3: ("OBFUSCATED", STATUS_WARNING),
    4: ("CONTEXTUAL", THREAT),
    5: ("APEX",       STATUS_MUTATION),
}


def _tier_bar(tier: int) -> str:
    """10-char tier bar — each tier = 2 blocks."""
    _, color = _TIER_LABELS.get(tier, ("?", FG_SECONDARY))
    filled = tier * 2
    empty  = (5 - tier) * 2
    return (
        f"[{color}]{'█' * filled}[/]"
        f"[{BORDER}]{'░' * empty}[/]"
    )


class BattlePanelWidget(Widget):
    """Battle engine — bypass-first hierarchy, Red vs Blue metrics."""

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {FG_SECONDARY}]BATTLE ENGINE[/]", id="bp-title")
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

        strat_label, strat_color = _TIER_LABELS.get(tier, ("?", FG_SECONDARY))

        block_rate = f"{blocks / attacks * 100:.0f}%" if attacks else "—"
        rate_color = STATUS_CLEAN if attacks and blocks / attacks > 0.9 else \
                     STATUS_WARNING if attacks and blocks / attacks > 0.5 else \
                     THREAT if attacks else FG_SECONDARY

        byp_color = THREAT if bypasses > 0 else FG_SECONDARY

        if bypasses > 0:
            headline = (
                f"\n[{THREAT} bold]⚠ {bypasses} BYPASS{'ES' if bypasses != 1 else ''}[/]  "
                f"[{rate_color}]{block_rate}[/][{FG_SECONDARY}] rate[/]  "
                f"[{FG_SECONDARY}]tick[/] [{FG_PRIMARY}]{tick}[/]  "
                f"[{FG_SECONDARY}]tier[/] [{strat_color}]{tier}/5[/]"
            )
        else:
            headline = (
                f"\n[{FG_SECONDARY}]BLOCK RATE[/]  "
                f"[{rate_color} bold]{block_rate}[/]  "
                f"[{FG_SECONDARY}]tick[/] [{FG_PRIMARY}]{tick}[/]  "
                f"[{FG_SECONDARY}]tier[/] [{strat_color}]{tier}/5 {strategy}[/]"
            )

        self.query_one("#bp-headline", Static).update(headline)

        self.query_one("#bp-tier-bar", Static).update(
            f"{_tier_bar(tier)}  [{strat_color}]{strat_label}[/]"
        )

        self.query_one("#bp-contest", Static).update(
            f"\n[{THREAT} bold]RED[/]                [{STATUS_CLEAN} bold]BLUE[/]\n"
            f"[{FG_SECONDARY}]attacks[/]  [{FG_PRIMARY} bold]{attacks:<8}[/]"
            f"[{FG_SECONDARY}]blocks [/]  [{STATUS_CLEAN} bold]{blocks}[/]\n"
            f"[{FG_SECONDARY}]bypass [/]  [{byp_color} bold]{bypasses:<8}[/]"
            f"[{FG_SECONDARY}]patches[/]  [{STATUS_MUTATION} bold]{patches}[/]"
        )

        result_color = STATUS_CLEAN if last_res == "BLOCKED" else \
                       THREAT if last_res == "BYPASSED" else FG_DIM
        self.query_one("#bp-last-result", Static).update(
            f"\n[{FG_SECONDARY}]LAST[/]  [{result_color} bold]{last_res}[/]"
        )

        if preview:
            trimmed = preview[:60] + "…" if len(preview) > 60 else preview
            self.query_one("#bp-preview", Static).update(
                f"\n[{FG_SECONDARY}]LAST ATTACK[/]\n"
                f"[{FG_DIM} italic]{trimmed}[/]"
            )

    def set_paused(self, paused: bool) -> None:
        """Visual indicator when paused."""
        if paused:
            self.query_one("#bp-title", Static).update(
                f"[bold {FG_SECONDARY}]BATTLE ENGINE[/]  [{STATUS_WARNING} bold]▌▌ PAUSED[/]"
            )
        else:
            self.query_one("#bp-title", Static).update(
                f"[bold {FG_SECONDARY}]BATTLE ENGINE[/]"
            )
