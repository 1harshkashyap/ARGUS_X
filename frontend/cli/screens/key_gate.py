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
        background: $background;
    }

    #gate-box {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 2 3;
        background: $surface;
    }

    #gate-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #gate-subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }

    #gate-error {
        color: $error;
        text-align: center;
        height: 1;
        margin-top: 1;
    }

    #key-input {
        margin-bottom: 1;
    }

    #submit-btn {
        width: 100%;
    }

    #gate-hint {
        text-align: center;
        color: $text-muted;
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
                        placeholder="AIzaSy...",
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

        if not (key.startswith("AI") and len(key) > 30):
            error_label.update(
                "Invalid format — Gemini keys start with 'AI' and are 35+ chars"
            )
            return

        # Valid — dismiss with the key
        self.dismiss(key)
