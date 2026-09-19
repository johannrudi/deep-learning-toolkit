"""Command-line entry point for the Deep Learning Toolkit."""

import typer

from dlk.nets import cli_efficientnet

app = typer.Typer(help="Deep Learning Toolkit command-line tools.")

nets_app = typer.Typer(help="Inspect network architectures.")
nets_app.add_typer(cli_efficientnet.app, name="efficientnet")
app.add_typer(nets_app, name="nets")


def main() -> None:
    """Run the Deep Learning Toolkit CLI."""
    app()


if __name__ == "__main__":
    main()
