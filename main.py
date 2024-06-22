from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Footer, Header, Input, RadioButton, RadioSet, Static

from lib import calculate_naive_bayes, process_tweet, train_naive_bayes


class DisplayWindow(Static):
    """Widget to display output text"""

    def append_text(self, text: str) -> None:
        self.update(self.renderable + "\n" + text)


class MainApp(App):
    # CSS_PATH = "styles.css"

    BINDINGS = [
        ("D", "toggle_dark_mode", "Toggle dark mode"),
        ("Q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()
        yield Container(
            Input(
                placeholder="Type your text here ...",
                max_length=100,
                id="text_input",
            ),
            RadioSet(
                RadioButton("Naive Bayes", value="naive_bayes", id="naive_bayes"),
                RadioButton(
                    "Logistic Regression",
                    value="logistic_regression",
                    id="logistic_regression",
                ),
            ),
            DisplayWindow(id="display_window"),
        )

    def on_mount(self) -> None:
        self.title = "Sentiment Analysis"
        self.sub_title = "Enter your text below and click 'Send' to analyze sentiment"

    def action_toggle_dark_mode(self) -> None:
        self.dark = not self.dark

    def action_quit(self) -> None:
        self.app.exit()

    @on(Input.Submitted, "#text_input")
    async def display_text(self) -> None:
        text = self.query_one("#text_input", Input).value
        display_window = self.query_one("#display_window", DisplayWindow)
        processed_text = process_tweet(text)

        display_window.update("")
        display_window.append_text(f"Original: {text}\nProcessed: {processed_text}\n")

        if self.query_one("#naive_bayes", RadioButton).value:
            if not Path("pretrained/nb_model.json").exists():
                display_window.append_text("Training Naive Bayes model ...\n")
                train_naive_bayes()
                display_window.append_text("Training completed\n")

            score = calculate_naive_bayes(text)
            sentiment = "😀" if score > 0 else ("😡" if score < 0 else "😶")

            display_window.append_text(f"Score: {score}\nSentiment: {sentiment}\n")
        elif self.query_one("#logistic_regression", RadioButton).value:
            # Placeholder for logistic regression implementation
            pass

        self.query_one("#text_input", Input).value = ""


if __name__ == "__main__":
    app = MainApp()
    app.run()
