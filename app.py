import typer
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, DataTable

app_cli = typer.Typer()

class Protagonist(App):
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),       # simplest option
        Binding("q", "quit", "Quit"),             # vim-style
        Binding("escape", "quit", "Quit"),        # intuitive for modal-style apps
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable()
        yield Footer()

@app_cli.command()
def dashboard(verbose: bool = False):
    """Launch the TUI dashboard."""
    Protagonist().run()

@app_cli.command()
def status():
    """Quick plain-text status (no TUI)."""
    print("All systems nominal")

if __name__ == "__main__":
    app_cli()