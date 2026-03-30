"""Rich-based terminal chat interface for LeanFormer."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
import requests


console = Console()


def chat(server_url: str = "http://localhost:8000"):
    console.print(Panel.fit(
        "[bold green]LeanFormer Chat[/bold green]\n"
        "Type your message. Type 'quit' to exit.\n"
        "Type '/stats' to see model efficiency stats.",
        border_style="green",
    ))

    history = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "/stats":
            _show_stats(server_url)
            continue

        history.append(f"User: {user_input}")
        prompt = "\n".join(history[-6:]) + "\nAssistant:"

        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            try:
                response = requests.post(
                    f"{server_url}/generate",
                    json={"prompt": prompt, "max_new_tokens": 300, "temperature": 0.8},
                    timeout=60,
                ).json()
            except requests.exceptions.ConnectionError:
                console.print("[red]Could not connect to server. Is it running?[/red]")
                continue

        response_text = response["text"].strip()
        history.append(f"Assistant: {response_text}")

        console.print(f"\n[bold green]Assistant[/bold green]")
        console.print(Markdown(response_text))

        # Show efficiency metrics inline
        console.print(
            f"\n[dim]{response['tokens_generated']} tokens "
            f"in {response['time_seconds']:.2f}s "
            f"({response['tokens_per_second']:.1f} tok/s) | "
            f"depth: {response['avg_exit_layer']:.1f}/{response['total_layers']} layers "
            f"({response['avg_depth_utilization']:.0%} utilization)[/dim]"
        )


def _show_stats(server_url: str):
    try:
        stats = requests.get(f"{server_url}/stats", timeout=10).json()
    except requests.exceptions.ConnectionError:
        console.print("[red]Could not connect to server.[/red]")
        return

    table = Table(title="LeanFormer Efficiency Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def main():
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    chat(url)


if __name__ == "__main__":
    main()
