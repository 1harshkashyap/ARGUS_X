"""
API Key Gate Screen — shown on startup if no key in .env.
Key is held in memory ONLY. Never written to disk by the TUI.
"""

from __future__ import annotations
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Input, Button, Label
from textual.containers import Center, Middle, Vertical
from textual import on


class KeyGateScreen(Screen[str]):
    """
    Full-screen key entry. Returns the key via self.dismiss(key).
    Parent app receives it via callback passed to push_screen.
    """

    CSS = """
    KeyGateScreen {
        align: center middle;
        background: #0a0a0c;
    }

    #gate-box {
        width: 62;
        height: auto;
        border: solid #4a9eff;
        padding: 2 4;
        background: #12121a;
    }

    #gate-title {
        text-align: center;
        text-style: bold;
        color: #4a9eff;
        margin-bottom: 1;
    }

    #gate-subtitle {
        text-align: center;
        color: #6b6b7a;
        margin-bottom: 2;
    }

    #gate-error {
        color: #ff4757;
        text-align: center;
        height: 1;
        margin-top: 1;
    }

    #key-input {
        margin-bottom: 1;
        border: solid #252530;
        background: #0a0a0c;
    }

    #key-input:focus {
        border: solid #4a9eff;
    }

    #submit-btn {
        width: 100%;
        background: #4a9eff;
        color: #ffffff;
        text-style: bold;
    }

    #submit-btn:hover {
        background: #3a8eef;
    }

    #gate-hint {
        text-align: center;
        color: #6b6b7a;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="gate-box"):
                    yield Static("ARGUS-X", id="gate-title")
                    yield Static(
                        "Defense Console — Enter your Gemini API key\n"
                        "Key stays in memory only — never written to disk",
                        id="gate-subtitle",
                    )
                    yield Input(
                        placeholder="AQ.Ab... or AIzaSy...",
                        password=True,
                        id="key-input",
                    )
                    yield Button("Continue →", variant="primary", id="submit-btn")
                    yield Label("", id="gate-error")
                    yield Static(
                        "Get a free key: aistudio.google.com",
                        id="gate-hint",
                    )

    def on_mount(self) -> None:
        self.query_one("#key-input", Input).focus()

    @on(Input.Submitted, "#key-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._try_submit(event.value)

    @on(Button.Pressed, "#submit-btn")
    def on_button_pressed(self) -> None:
        key = self.query_one("#key-input", Input).value
        self._try_submit(key)

    def _try_submit(self, key: str) -> None:
        key = key.strip()
        error_label = self.query_one("#gate-error", Label)

        if not key:
            error_label.update("Please enter your API key")
            return

        # Accept both old format (AIza...) and new format (AQ.Ab...)
        if len(key) < 20:
            error_label.update(
                "Invalid format — key is too short (expected 20+ characters)"
            )
            return

        # Valid — dismiss with the key
        self.dismiss(key)
