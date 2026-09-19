"""Command-line inspection tool for EfficientNet1D stage configs and parameter counts."""

import inspect
from collections.abc import Callable
from typing import Annotated, Literal

import torch.nn as nn
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from rich.tree import Tree

from dlk.nets import efficientnet1d
from dlk.nets.efficientnet1d import StageSpec

app = typer.Typer(
    help="Inspect EfficientNet1D stage configurations and parameter counts."
)
console = Console()
# progress goes to stderr so `--json` output on stdout stays valid JSON when piped
progress_console = Console(stderr=True)

Family = Literal["v1", "v2"]
SortBy = Literal["name", "blocks", "params"]


def _family_of(name: str) -> Family:
    """Classify a builder or model class name as EfficientNetV1 or V2.

    Args:
        name: A `get_efficientnet_*_config` function name or `EfficientNet*` class name.

    Returns:
        Family: `"v1"` or `"v2"`.
    """
    if "V1" in name or "_v1_" in name:
        return "v1"
    if "V2" in name or "_v2_" in name:
        return "v2"
    raise ValueError(f"cannot determine architecture family for {name!r}")


def _discover_config_builders() -> list[tuple[str, Callable[[], list[StageSpec]]]]:
    """Find zero-argument `get_efficientnet_*` stage config builders in `efficientnet1d`.

    Returns:
        list[tuple[str, Callable[[], list[StageSpec]]]]: Name and builder pairs,
        sorted alphabetically.
    """
    return sorted(
        (name, obj)
        for name, obj in inspect.getmembers(efficientnet1d, inspect.isfunction)
        if name.startswith("get_efficientnet_")
        and obj.__module__ == efficientnet1d.__name__
    )


def _discover_model_classes() -> list[tuple[str, type[nn.Module]]]:
    """Find concrete `EfficientNet*` model classes in `efficientnet1d`.

    Excludes `ScalableEfficientNet1D`, the base class the concrete models subclass.

    Returns:
        list[tuple[str, type[nn.Module]]]: Name and class pairs, sorted alphabetically.
    """
    return sorted(
        (name, obj)
        for name, obj in inspect.getmembers(efficientnet1d, inspect.isclass)
        if obj.__module__ == efficientnet1d.__name__
        and issubclass(obj, nn.Module)
        and name.startswith("EfficientNet")
        and name != "ScalableEfficientNet1D"
    )


def _format_param_count(num_params: int) -> str:
    """Format a parameter count with a compact magnitude suffix.

    Args:
        num_params: Total number of trainable parameters.

    Returns:
        str: Human-readable parameter count, e.g. `"3.87M"` or `"72.9K"`.
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def _magnitude_style(num_params: int) -> str:
    """Pick a rich color for a parameter count, by magnitude band.

    Args:
        num_params: Total number of trainable parameters.

    Returns:
        str: Rich color name.
    """
    if num_params < 1_000_000:
        return "green"
    if num_params < 50_000_000:
        return "yellow"
    return "red"


def _count_model_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: Instantiated module.

    Returns:
        int: Sum of `numel()` over parameters with `requires_grad`.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _render_stage_table(name: str, stages: list[StageSpec]) -> Table:
    """Render one stage config builder's stages as a rich table.

    Args:
        name: Builder function name, used as the table title.
        stages: Ordered stage specifications returned by the builder.

    Returns:
        Table: One row per stage.
    """
    total_blocks = sum(spec.config.num_layers for spec in stages)
    table = Table(
        title=f"{name}()  [{len(stages)} stages, {total_blocks} blocks]",
        title_justify="left",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Block")
    table.add_column("K", justify="right")
    table.add_column("S", justify="right")
    table.add_column("E", justify="right")
    table.add_column("Cin", justify="right")
    table.add_column("Cout", justify="right")
    table.add_column("Rep", justify="right")
    table.add_column("SE", justify="right")
    for i, stage in enumerate(stages, start=1):
        config = stage.config
        se = f"{config.se_ratio:g}" if config.se_ratio is not None else "—"
        table.add_row(
            f"{i:02d}",
            stage.block_cls.__name__,
            str(config.kernel_size),
            str(config.stride),
            str(config.expand_ratio),
            str(config.input_channels),
            str(config.output_channels),
            str(config.num_layers),
            se,
        )
    return table


def _render_stage_tree(name: str, stages: list[StageSpec]) -> Tree:
    """Render one stage config builder's per-block expansion as a rich tree.

    Unlike the table view (one row per stage), this expands each stage into
    its actual per-block channel transitions, mirroring the block-construction
    loop in `ScalableEfficientNet1D.__init__`: only the first block in a stage
    uses the stage's stride and input channels, later blocks use stride 1 and
    carry the stage's output channels through unchanged.

    Args:
        name: Builder function name, used as the tree root label.
        stages: Ordered stage specifications returned by the builder.

    Returns:
        Tree: One branch per stage, one leaf per block.
    """
    total_blocks = sum(spec.config.num_layers for spec in stages)
    tree = Tree(f"[bold]{name}()[/bold]  [{len(stages)} stages, {total_blocks} blocks]")
    for stage_num, stage in enumerate(stages, start=1):
        config = stage.config
        se = f"se{config.se_ratio:g}" if config.se_ratio is not None else "se—"
        num_blocks = config.num_layers
        branch = tree.add(
            f"Stage {stage_num:02d}  {stage.block_cls.__name__}  "
            f"k{config.kernel_size} e{config.expand_ratio} {se}  "
            f"({num_blocks} block{'s' if num_blocks != 1 else ''})"
        )
        for block_num in range(1, num_blocks + 1):
            if block_num == 1:
                stride, input_channels = config.stride, config.input_channels
            else:
                stride, input_channels = 1, config.output_channels
            branch.add(
                f"block {block_num}  s{stride}  {input_channels}→{config.output_channels}"
            )
    return tree


def _render_params_table(family_name: str, rows: list[dict[str, object]]) -> Table:
    """Render one family's parameter-count rows as a rich table.

    Args:
        family_name: `"v1"` or `"v2"`, used as the table title.
        rows: Records with `model`, `blocks`, and `params` keys.

    Returns:
        Table: One row per model, magnitude-colored.
    """
    table = Table(
        title=f"EfficientNet {family_name.upper()} family",
        title_justify="left",
    )
    table.add_column("Model")
    table.add_column("Blocks", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Raw", justify="right")
    for row in rows:
        num_params = row["params"]
        assert isinstance(num_params, int)
        style = _magnitude_style(num_params)
        table.add_row(
            str(row["model"]),
            str(row["blocks"]),
            f"[{style}]{_format_param_count(num_params)}[/{style}]",
            f"{num_params:,}",
        )
    return table


def _scaling_cell(base_value: int, scaled_value: int) -> str:
    """Format a table cell contrasting a base value with its compound-scaled result.

    Args:
        base_value: Value before width/depth scaling.
        scaled_value: Value after width/depth scaling.

    Returns:
        str: The bare value if scaling left it unchanged, otherwise a
        highlighted `base → scaled` arrow.
    """
    if base_value == scaled_value:
        return str(base_value)
    return f"[bold cyan]{base_value} → {scaled_value}[/bold cyan]"


def _resolve_scaled_channels(module: nn.Module | None, fallback: int) -> int:
    """Read the actual scaled channel count built into a stem or head submodule.

    `ScalableEfficientNet1D` builds `stem` and `head` as an `nn.Sequential` whose
    first layer is the width-scaled `nn.Conv1d`; reading it directly avoids
    recomputing `round_filters` with the (unstored) `depth_divisor`/`min_depth`.

    Args:
        module: A model's `stem` or `head` submodule, or `None`.
        fallback: Unscaled channel count to report if the module has no such
            conv layer to inspect (e.g. the stem or head is disabled).

    Returns:
        int: The scaled channel count.
    """
    if isinstance(module, nn.Sequential):
        first_layer = module[0]
        if isinstance(first_layer, nn.Conv1d):
            return first_layer.out_channels
    return fallback


def _render_scaling_table(
    base_stages: list[StageSpec], scaled_stages: list[StageSpec]
) -> Table:
    """Render a model's stages with base-to-scaled compound-scaling deltas.

    Args:
        base_stages: Unscaled stage specs, as returned by the matching
            `get_efficientnet_*` builder.
        scaled_stages: The model's actual (width/depth-scaled) stage specs.

    Returns:
        Table: One row per stage; `Cin`/`Cout`/`Rep` show `base → scaled` where
        compound scaling changed the value.
    """
    table = Table()
    table.add_column("#", justify="right", style="dim")
    table.add_column("Block")
    table.add_column("K", justify="right")
    table.add_column("S", justify="right")
    table.add_column("E", justify="right")
    table.add_column("Cin", justify="right")
    table.add_column("Cout", justify="right")
    table.add_column("Rep", justify="right")
    table.add_column("SE", justify="right")
    for i, (base, scaled) in enumerate(zip(base_stages, scaled_stages), start=1):
        base_cfg, scaled_cfg = base.config, scaled.config
        se = f"{scaled_cfg.se_ratio:g}" if scaled_cfg.se_ratio is not None else "—"
        table.add_row(
            f"{i:02d}",
            scaled.block_cls.__name__,
            str(scaled_cfg.kernel_size),
            str(scaled_cfg.stride),
            str(scaled_cfg.expand_ratio),
            _scaling_cell(base_cfg.input_channels, scaled_cfg.input_channels),
            _scaling_cell(base_cfg.output_channels, scaled_cfg.output_channels),
            _scaling_cell(base_cfg.num_layers, scaled_cfg.num_layers),
            se,
        )
    return table


def _print_model_scaling(model_name: str) -> None:
    """Print one concrete model's stage configs with compound-scaling detail.

    Compares the model's actual (width/depth-scaled) `stage_specs`, `stem`, and
    `head` against the unscaled base values `ScalableEfficientNet1D.__init__`
    stores on the instance.

    Args:
        model_name: A concrete `EfficientNet*` class name.

    Raises:
        typer.Exit: If `model_name` does not match a discovered model class.
    """
    model_classes = dict(_discover_model_classes())
    if model_name not in model_classes:
        console.print(
            f"[red]Unknown model {model_name!r}.[/red] Run `params` to list model names."
        )
        raise typer.Exit(code=1)

    instance = model_classes[model_name](input_length=None)
    width_coefficient = getattr(instance, "width_coefficient", None)
    depth_coefficient = getattr(instance, "depth_coefficient", None)
    base_stem_channels = getattr(instance, "base_stem_channels", None)
    base_head_channels = getattr(instance, "base_head_channels", None)
    base_stage_specs = getattr(instance, "base_stage_specs", None)
    scaled_stage_specs = getattr(instance, "stage_specs", None)
    stem_module = getattr(instance, "stem", None)
    head_module = getattr(instance, "head", None)
    assert isinstance(width_coefficient, float)
    assert isinstance(depth_coefficient, float)
    assert isinstance(base_stem_channels, int)
    assert isinstance(base_head_channels, int)
    assert isinstance(base_stage_specs, list)
    assert isinstance(scaled_stage_specs, list)

    stem_scaled = _resolve_scaled_channels(stem_module, base_stem_channels)
    head_scaled = _resolve_scaled_channels(head_module, base_head_channels)

    console.print(
        f"[bold]{model_name}[/bold]  "
        f"(width={width_coefficient:g}, depth={depth_coefficient:g})"
    )
    console.print(
        f"stem   {_scaling_cell(base_stem_channels, stem_scaled)}"
        f"      head  {_scaling_cell(base_head_channels, head_scaled)}"
    )
    console.print()
    console.print(_render_scaling_table(base_stage_specs, scaled_stage_specs))


@app.command()
def configs(
    family: Annotated[
        Family | None, typer.Option(help="Restrict to one architecture family.")
    ] = None,
    tree: Annotated[
        bool, typer.Option(help="Render stages as a tree instead of a table.")
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            help=(
                "Show compound-scaling detail for one concrete EfficientNet* model: "
                "base vs. width/depth-scaled Cin/Cout/Rep, stem, and head. "
                "Overrides --family/--tree."
            )
        ),
    ] = None,
) -> None:
    """Print stage configurations for every `get_efficientnet_*` builder."""
    if model is not None:
        _print_model_scaling(model)
        return

    builders = _discover_config_builders()
    if family is not None:
        builders = [(name, obj) for name, obj in builders if _family_of(name) == family]

    for name, builder in builders:
        stages = builder()
        if not stages:
            console.print(f"{name}()  (empty)")
            continue
        renderable = (
            _render_stage_tree(name, stages)
            if tree
            else _render_stage_table(name, stages)
        )
        console.print(renderable)
        console.print()


@app.command()
def params(
    family: Annotated[
        Family | None, typer.Option(help="Restrict to one architecture family.")
    ] = None,
    sort_by: Annotated[
        SortBy, typer.Option(help="Sort key for the ranking within each family.")
    ] = "params",
    input_channels: Annotated[
        int, typer.Option(help="Channel count passed to each model constructor.")
    ] = 1,
    num_classes: Annotated[
        int, typer.Option(help="Class count passed to each model constructor.")
    ] = 2,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON instead of tables."),
    ] = False,
) -> None:
    """Instantiate every `EfficientNet*` model and print its parameter count."""
    model_classes = _discover_model_classes()
    if family is not None:
        model_classes = [
            (name, cls) for name, cls in model_classes if _family_of(name) == family
        ]

    records: list[dict[str, object]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=progress_console,
        transient=True,
    ) as progress:
        task = progress.add_task("Instantiating models...", total=len(model_classes))
        for name, model_cls in model_classes:
            progress.update(task, description=f"Instantiating {name}...")
            model = model_cls(
                input_channels=input_channels,
                input_length=None,
                num_classes=num_classes,
            )
            num_params = _count_model_parameters(model)
            blocks = getattr(model, "blocks", None)
            num_blocks = len(blocks) if isinstance(blocks, nn.ModuleList) else 0
            records.append(
                {
                    "model": name,
                    "family": _family_of(name),
                    "blocks": num_blocks,
                    "params": num_params,
                }
            )
            progress.advance(task)

    if json_output:
        console.print_json(data=records)
        return

    sort_key: Callable[[dict[str, object]], object] = {
        "name": lambda r: r["model"],
        "blocks": lambda r: r["blocks"],
        "params": lambda r: r["params"],
    }[sort_by]

    families: list[Family] = [family] if family is not None else ["v1", "v2"]
    for fam in families:
        rows = sorted((r for r in records if r["family"] == fam), key=sort_key)
        console.print(_render_params_table(fam, rows))
        console.print()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Inspect EfficientNet1D stage configurations and parameter counts.

    With no subcommand, prints both `configs` and `params` with default settings.
    """
    if ctx.invoked_subcommand is None:
        configs()
        params()


if __name__ == "__main__":
    app()
