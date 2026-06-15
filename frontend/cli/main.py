"""
ARGUS-X Defense Console — Full TUI Application
Run: python main.py
"""
from __future__ import annotations
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import TabbedContent

from screens.key_gate import KeyGateScreen
from widgets.stats_bar    import StatsBar
from widgets.event_list   import EventListWidget, EventSelected
from widgets.xai_panel    import XAIPanelWidget
from widgets.battle_panel import BattlePanelWidget
from widgets.input_panel  import InputPanelWidget, ChatRequest
from config import _ENV_GEMINI_KEY, STATS_POLL_INTERVAL, \
                   EVENTS_POLL_INTERVAL, BATTLE_POLL_INTERVAL
import api


class ArgusApp(App):
    """ARGUS-X Defense Console — production-grade security TUI."""

    CSS_PATH = "app.tcss"
    TITLE    = "ARGUS-X"
    SUB_TITLE = "Defense Console"

    BINDINGS = [
        Binding("q",    "quit",          "Quit",         show=True),
        Binding("s",    "simulate",      "Simulate",     show=True),
        Binding("p",    "toggle_pause",  "Pause/Resume", show=True),
        Binding("r",    "refresh_all",   "Refresh",      show=True),
        Binding("1",    "focus_chat",    "Chat",         show=False),
        Binding("2",    "focus_redteam", "Red Team",     show=False),
        Binding("f1",   "preset_1",      "F1 Inject",    show=False),
        Binding("f2",   "preset_2",      "F2 Jailbreak", show=False),
        Binding("f3",   "preset_3",      "F3 Exfil",     show=False),
        Binding("f4",   "preset_4",      "F4 Role",      show=False),
        Binding("f5",   "preset_5",      "F5 Indirect",  show=False),
        Binding("f6",   "preset_6",      "F6 APEX",      show=False),
        Binding("up",   "history_up",    "History ↑",    show=False),
        Binding("down", "history_down",  "History ↓",    show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._api_key:   str  = ""
        self._is_paused: bool = False
        self._chat_busy: bool = False

    # ── Composition ───────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield StatsBar(id="stats-bar")
        with Horizontal(id="main-content"):
            yield EventListWidget(id="event-list")
            yield XAIPanelWidget(id="xai-panel")
            yield BattlePanelWidget(id="battle-panel")
        yield InputPanelWidget(id="input-panel")

    # ── Startup ───────────────────────────────────────────────────────

    def on_mount(self) -> None:
        if _ENV_GEMINI_KEY and len(_ENV_GEMINI_KEY) > 30:
            self._api_key = _ENV_GEMINI_KEY
            self._post_key_setup()
        else:
            self.push_screen(KeyGateScreen(), self._on_key_received)

    def _on_key_received(self, key: str) -> None:
        self._api_key = key
        self._post_key_setup()

    def _post_key_setup(self) -> None:
        panel = self.query_one(InputPanelWidget)
        panel.set_api_key(self._api_key)
        self.call_after_refresh(self._start_polling)

    async def _start_polling(self) -> None:
        await self._refresh_health()
        await self._refresh_stats()
        await self._refresh_events()
        await self._refresh_battle()
        self.set_interval(STATS_POLL_INTERVAL,  self._refresh_stats)
        self.set_interval(EVENTS_POLL_INTERVAL, self._refresh_events)
        self.set_interval(BATTLE_POLL_INTERVAL, self._refresh_battle)

    # ── Polling ───────────────────────────────────────────────────────

    async def _refresh_health(self) -> None:
        result = await api.health()
        bar = self.query_one(StatsBar)
        bar.set_online(isinstance(result, api.Ok))

    async def _refresh_stats(self) -> None:
        result = await api.stats()
        if isinstance(result, api.Ok):
            data = result.data
            bar  = self.query_one(StatsBar)
            # Merge service info from health into stats
            health_res = await api.health()
            if isinstance(health_res, api.Ok):
                data["services"] = health_res.data.get("services", {})
            bar.update_stats(data)

    async def _refresh_events(self) -> None:
        result = await api.logs(limit=50)
        if isinstance(result, api.Ok):
            events = result.data.get("events", [])
            self.query_one(EventListWidget).load_events(events)

    async def _refresh_battle(self) -> None:
        result = await api.battle_state()
        if isinstance(result, api.Ok):
            self.query_one(BattlePanelWidget).update_state(result.data)

    # ── Event handling ────────────────────────────────────────────────

    def on_event_selected(self, message: EventSelected) -> None:
        self.query_one(XAIPanelWidget).show_event(message.event)

    def on_chat_request(self, message: ChatRequest) -> None:
        self.run_worker(
            self._execute_chat(
                message.message, message.session_id, message.api_key
            ),
            exclusive=False,
        )

    async def _execute_chat(
        self, message: str, session_id: str, api_key: str
    ) -> None:
        panel = self.query_one(InputPanelWidget)
        result = await api.chat(message, session_id, api_key)

        if isinstance(result, api.Ok):
            data = result.data

            # Determine which tab is active — route response accordingly
            tabs = panel.query_one(TabbedContent)
            active = tabs.active

            if active == "chat":
                panel.display_chat_response(data)
            else:
                panel.display_attack_response(data)

            # Update XAI panel with new event analysis
            if data.get("blocked"):
                self.query_one(XAIPanelWidget).show_event(data)
                # Notify about campaign-level threats
                session_level = data.get("session_threat_level", "LOW")
                if session_level == "CRITICAL":
                    self.notify(
                        "Session reached CRITICAL threat level!",
                        severity="error",
                        timeout=5,
                    )
                elif session_level == "HIGH":
                    self.notify(
                        "Session threat level: HIGH",
                        severity="warning",
                        timeout=3,
                    )

            # Refresh events + battle after any chat request
            await self._refresh_events()
            await self._refresh_battle()
            await self._refresh_stats()

        else:
            error = result.message
            panel.display_chat_response({
                "blocked": False,
                "response": f"Error: {error}",
                "latency_ms": 0,
            })

        panel.set_busy(False)

    # ── Keyboard actions ──────────────────────────────────────────────

    async def action_simulate(self) -> None:
        self.notify("Forcing battle cycle...", timeout=2)
        result = await api.cycle()
        if isinstance(result, api.Ok):
            state = result.data
            self.query_one(BattlePanelWidget).update_state(state)
            last = state.get("last_attack_result", "?")
            self.notify(
                f"Cycle complete — {last}  tick={state.get('tick')}",
                severity="information" if last == "BLOCKED" else "warning",
                timeout=4,
            )
            await self._refresh_events()
        else:
            self.notify(f"Cycle failed: {result.message}", severity="error")

    async def action_toggle_pause(self) -> None:
        panel = self.query_one(BattlePanelWidget)
        if self._is_paused:
            result = await api.resume()
            if isinstance(result, api.Ok):
                self._is_paused = False
                panel.set_paused(False)
                self.notify("Battle engine resumed", severity="information")
        else:
            result = await api.pause()
            if isinstance(result, api.Ok):
                self._is_paused = True
                panel.set_paused(True)
                self.notify("Battle engine paused", severity="warning")

    async def action_refresh_all(self) -> None:
        await self._refresh_health()
        await self._refresh_stats()
        await self._refresh_events()
        await self._refresh_battle()
        self.notify("Refreshed", timeout=1)

    def action_focus_chat(self) -> None:
        panel = self.query_one(InputPanelWidget)
        tabs  = panel.query_one(TabbedContent)
        tabs.active = "chat"

    def action_focus_redteam(self) -> None:
        panel = self.query_one(InputPanelWidget)
        tabs  = panel.query_one(TabbedContent)
        tabs.active = "redteam"

    def _fire_preset(self, preset_name: str) -> None:
        panel = self.query_one(InputPanelWidget)
        tabs  = panel.query_one(TabbedContent)
        tabs.active = "redteam"
        panel.fire_preset(preset_name)

    def action_preset_1(self) -> None: self._fire_preset("F1 Inject")
    def action_preset_2(self) -> None: self._fire_preset("F2 Jailbreak")
    def action_preset_3(self) -> None: self._fire_preset("F3 Exfil")
    def action_preset_4(self) -> None: self._fire_preset("F4 Role")
    def action_preset_5(self) -> None: self._fire_preset("F5 Indirect")
    def action_preset_6(self) -> None: self._fire_preset("F6 APEX")

    def action_history_up(self) -> None:
        self.query_one(InputPanelWidget).history_up()

    def action_history_down(self) -> None:
        self.query_one(InputPanelWidget).history_down()


if __name__ == "__main__":
    ArgusApp().run()
