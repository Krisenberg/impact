from __future__ import annotations

import dataclasses
from enum import Enum
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

import click

from impact.config import (
    ACTRConfig,
    DatasetConfig,
    EvalConfig,
    GTSConfig,
    ImpactConfig,
    SASRecConfig,
    SequenceConfig,
    SessionConfig,
    SystemConfig,
)
from impact.dataset import DatasetSize, DatasetType, InteractionType
from impact.utils import find_sibling_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import StrEnum


def inject_impact_config(f: Callable[..., Any]) -> Callable[..., Any]:
    """Inject ImpactConfig fields as Click options and pass the instantiated config.

    Adds one Click option per ImpactConfig field (grouped by nested config section),
    using ImpactConfig's defaults as the option defaults. The decorated function must
    accept `impact_config: ImpactConfig` as a keyword argument; it is constructed at
    call time from the parsed CLI values.

    Use this for commands that configure a fresh run from scratch (e.g. training, grid
    search). For commands operating on an existing checkpoint (e.g. evaluation), use
    `inject_config_from_checkpoint` instead, which loads the config that was saved
    alongside the checkpoint rather than rebuilding it from CLI flags.
    """
    # Define all click options, matching the defaults from ImpactConfig
    options = [
        click.option(
            "--dataset-size",
            type=click.Choice([e.value for e in DatasetSize]),
            default=DatasetSize.SIZE_50M.value,
            show_default=True,
            help="Yambda dataset variant to process (50M, 500M, or 5B).",
        ),
        click.option(
            "--dataset-format",
            type=click.Choice([DatasetType.FLAT.value]),
            default=DatasetType.FLAT.value,
            show_default=True,
            help="Yambda dataset format type. Currently only FLAT is supported.",
        ),
        click.option(
            "--dataset-interaction-type",
            type=click.Choice([e.value for e in InteractionType]),
            default=InteractionType.MULTI_EVENT.value,
            show_default=True,
            help="""
                Type of interactions (events) to load from the Yambda dataset.
                MULTI_EVENT includes all event types (listen, like, etc.); other
                variants contain only a specific subset.
            """,
        ),
        click.option(
            "--session-tracks-per-session",
            type=int,
            default=10,
            show_default=True,
            help="""
                Number of tracks that constitute a session. Denoted as K in the thesis.
                Sessions with less than K tracks are discarded. Sessions with more than
                K tracks are truncated.
            """,
        ),
        click.option(
            "--sequence-context-sessions",
            type=int,
            default=20,
            show_default=True,
            help="""
                Number of historical sessions used as context. Denoted as L in the
                thesis.
            """,
        ),
        click.option(
            "--sequence-step",
            type=int,
            default=5,
            show_default=True,
            help="Step size for the sliding window sequence extraction.",
        ),
        click.option(
            "--gts-test-days",
            type=float,
            default=1.0,
            show_default=True,
            help="""
                Number of days at the end of the timeline allocated for the test set.
            """,
        ),
        click.option(
            "--gts-val-days",
            type=float,
            default=1.0,
            show_default=True,
            help="""
                Number of days preceding test set allocated for the validation set.
            """,
        ),
        click.option(
            "--gts-gap-minutes",
            type=float,
            default=30.0,
            show_default=True,
            help="""
                Buffer time (in minutes) between dataset splits. The purpose of this
                parameter is to mimic production system deployment latency.
            """,
        ),
        click.option(
            "--actr-decay",
            type=float,
            default=0.5,
            show_default=True,
            help="""
                Power-law time decay exponent. Denoted as d in the thesis. Higher
                values result in older listens being forgotten faster.
            """,
        ),
        click.option(
            "--actr-organic-weight",
            type=float,
            default=1.0,
            show_default=True,
            help="Contribution weight for organic (user-initiated) listens.",
        ),
        click.option(
            "--actr-autoplay-weight",
            type=float,
            default=0.5,
            show_default=True,
            help="Contribution weight for autoplay (recsys-initiated) listens.",
        ),
        click.option(
            "--actr-eps",
            type=float,
            default=1e-8,
            show_default=True,
            help="Numerical floor added before applying log to prevent log(0).",
        ),
        click.option(
            "--sasrec-embedding-dim",
            type=int,
            default=64,
            show_default=True,
            help="Dimension of item and positional embeddings.",
        ),
        click.option(
            "--sasrec-num-heads",
            type=int,
            default=2,
            show_default=True,
            help="Number of attention heads in each Transformer layer.",
        ),
        click.option(
            "--sasrec-num-layers",
            type=int,
            default=2,
            show_default=True,
            help="Number of Transformer encoder layers.",
        ),
        click.option(
            "--sasrec-dropout-rate",
            type=float,
            default=0.2,
            show_default=True,
            help="Dropout probability applied after embedding within each layer.",
        ),
        click.option(
            "--sasrec-train-batch-size",
            type=int,
            default=2048,
            show_default=True,
            help="Number of sequences per training batch.",
        ),
        click.option(
            "--sasrec-eval-batch-size",
            type=int,
            default=2048,
            show_default=True,
            help="""
                Number of sequences per evaluation batch.

                For larger dataset variants, should be lower than training batch size
                due to full-catalog scoring memory requirements.
            """,
        ),
        click.option(
            "--sasrec-max-epochs",
            type=int,
            default=100,
            show_default=True,
            help="Maximum number of training epochs to run.",
        ),
        click.option(
            "--sasrec-learning-rate",
            type=float,
            default=0.001,
            show_default=True,
            help="Learning rate for the optimizer.",
        ),
        click.option(
            "--sasrec-val-k-cutoff",
            type=int,
            default=10,
            show_default=True,
            help="""
                Rank cutoff for NDCG@K and Recall@K metrics used during model training.
            """,
        ),
        click.option(
            "--sasrec-patience",
            type=int,
            default=10,
            show_default=True,
            help="""
                Number of epochs to wait for validation improvement before stopping.
            """,
        ),
        click.option(
            "--eval-batch-size",
            type=int,
            default=512,
            show_default=True,
            help="""
                Number of sequences per evaluation batch. Should be lower than training
                batch size due to full-catalog scoring memory requirements.
            """,
        ),
        click.option(
            "--eval-k-cutoffs",
            type=tuple[int, ...],
            default=(5, 10, 40, 100),
            show_default=True,
            help="Rank cutoffs for computing NDCG@K and Recall@K during evaluation.",
        ),
        click.option(
            "--eval-alpha",
            type=float,
            default=0.5,
            show_default=True,
            help="""
                Relative contribution of SASRec and ACT-R in the final late-fusion
                score. The final score is computed as: alpha * SASRec + (1 - alpha) *
                ACT-R, where 1.0 corresponds to SASRec-only recommendations and 0.0 to
                ACT-R-only recommendations.
            """,
        ),
        click.option(
            "--system-seed",
            type=int,
            default=42,
            show_default=True,
            help="Global random seed for strict reproducibility.",
        ),
        click.option(
            "--system-num-workers",
            type=int,
            default=4,
            show_default=True,
            help="""
                Number of subprocesses to use for data loading. Set to 0 to load data in
                the main process, or higher for parallel loading (hardware-dependent).
            """,
        ),
    ]

    # Apply Click options to the function in reverse order to maintain defined ordering
    for option in reversed(options):
        f = option(f)

    @wraps(f)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        config_section_by_prefix: dict[str, _ConfigSection] = {
            "dataset_": _ConfigSection(DatasetConfig),
            "session_": _ConfigSection(SessionConfig),
            "sequence_": _ConfigSection(SequenceConfig),
            "gts_": _ConfigSection(GTSConfig),
            "actr_": _ConfigSection(ACTRConfig),
            "sasrec_": _ConfigSection(SASRecConfig),
            "eval_": _ConfigSection(EvalConfig),
            "system_": _ConfigSection(SystemConfig),
        }
        enum_by_section_prefix_field_name: dict[tuple[str, str], type[StrEnum]] = {
            ("dataset_", "size"): DatasetSize,
            ("dataset_", "format"): DatasetType,
            ("dataset_", "interaction_type"): InteractionType,
        }
        remaining_kwargs: dict[str, Any] = {}

        for key, value in kwargs.items():
            matched = False
            for prefix, config_section in config_section_by_prefix.items():
                if not key.startswith(prefix):
                    continue

                nested_key = key.removeprefix(prefix)
                if nested_key not in config_section.field_names:
                    continue

                if (prefix, nested_key) in enum_by_section_prefix_field_name:
                    # Parse string choice back into its respective Enum class
                    enum = enum_by_section_prefix_field_name[(prefix, nested_key)]
                    config_section.kwargs[nested_key] = enum(value)
                else:
                    config_section.kwargs[nested_key] = value

                matched = True
                break

            if not matched:
                remaining_kwargs[key] = value

        impact_config = ImpactConfig(
            **{
                prefix.removesuffix("_"): config_section.cls(**config_section.kwargs)
                for prefix, config_section in config_section_by_prefix.items()
            }
        )

        # Inject the config object and forward remaining CLI options
        return f(*args, impact_config=impact_config, **remaining_kwargs)

    return _wrapper


@dataclasses.dataclass
class _ConfigSection:
    """Metadata and collected CLI arguments used to construct ImpactConfig section."""

    cls: type
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @cached_property
    def field_names(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self.cls)]

    @cached_property
    def type_hints(self) -> dict[str, Any]:
        return get_type_hints(self.cls)


def inject_impact_config_from_checkpoint(f: Callable[..., Any]) -> Callable[..., Any]:
    """Add Click option for SASRecNBR checkpoint path and load its sibling YAML config.

    Adds a required `--sasrec-checkpoint-path` option (validated to exist).
    The decorated function must accept both of the following keyword arguments, which
    this decorator provides:
    - `sasrec_checkpoint_path: Path`: path to the checkpoint parsed from the CLI option.
    - `impact_config: ImpactConfig`: config loaded via `utils.find_sibling_config()` and
        `ImpactConfig.from_yaml()` from the checkpoint's sibling config file.

    Use for commands operating on an existing trained model (e.g., IMPACT evaluation
    or plotting). Combine with `cli.inject_impact_config_overrides(...)` to allow CLI
    overrides of specific config sections (e.g. eval, actr) on top of the loaded config.
    For commands that build a fresh config from CLI flags, use `inject_impact_config`
    instead.
    """
    options = [
        click.option(
            "--sasrec-checkpoint-path",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Path to a trained SASRecNBR checkpoint that will be evaluated.",
        ),
    ]

    # Apply Click options to the function in reverse order to maintain defined ordering
    for option in reversed(options):
        f = option(f)

    @wraps(f)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        checkpoint_path: Path = kwargs["sasrec_checkpoint_path"]
        kwargs["impact_config"] = ImpactConfig.from_yaml(
            path=find_sibling_config(checkpoint_path)
        )
        return f(*args, **kwargs)

    return _wrapper


_OVERRIDE_SECTIONS: dict[str, _ConfigSection] = {
    "actr": _ConfigSection(ACTRConfig),
    "eval": _ConfigSection(EvalConfig),
    "system": _ConfigSection(SystemConfig),
}


def inject_impact_config_overrides(
    *sections: Literal["actr", "eval", "system"],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Add CLI options to override fields of selected ImpactConfig sections.

    Args:
        sections: Section names to expose as overrides, e.g. "eval" or "actr". Each
            field of the corresponding config dataclass becomes a `--{section}-{field}`
            option, defaulting to None (meaning "keep the value from loaded config").

    The decorated function must accept `impact_config: ImpactConfig` as a keyword
    argument, populated upstream. Only explicitly provided CLI values override the
    loaded configuration. All other fields retain their values from the checkpoint
    snapshot.

    Use together with `cli.inject_impact_config_from_checkpoint()` for commands that
    load a checkpoint's saved config but need to adjust evaluation/runtime parameters.
    """

    def _decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        unsupported = set(sections) - set(_OVERRIDE_SECTIONS.keys())
        if unsupported:
            raise ValueError(f"Unsupported override sections: {unsupported}")

        chosen = {s: _OVERRIDE_SECTIONS[s] for s in sections}
        options: list[Callable[[Any], Any]] = []
        for section, config_section in chosen.items():
            for field_name in config_section.field_names:
                field_type = config_section.type_hints[field_name]

                option_name = f"--{section}-{field_name}".replace("_", "-")
                multiple = get_origin(field_type) is tuple

                options.append(
                    click.option(
                        option_name,
                        type=_click_type_for(field_type),
                        multiple=multiple,
                        default=() if multiple else None,
                        help=(
                            f"Override {config_section.cls.__name__}.{field_name}. "
                            "Defaults to the loaded config's value."
                        ),
                    )
                )

        # Apply Click options to the function in reverse order to maintain ordering
        for option in reversed(options):
            f = option(f)

        @wraps(f)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            impact_config: ImpactConfig = kwargs["impact_config"]

            for section, config_section in chosen.items():
                overrides: dict[str, Any] = {}
                for field_name in config_section.field_names:
                    kw_key = f"{section}_{field_name}"
                    value = kwargs.pop(kw_key, None)
                    if value in (None, ()):
                        continue
                    overrides[field_name] = value

                if overrides:
                    current_section = getattr(impact_config, section)
                    impact_config = dataclasses.replace(
                        impact_config,
                        **{section: dataclasses.replace(current_section, **overrides)},
                    )

            kwargs["impact_config"] = impact_config
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


def _click_type_for(type_: Any) -> Any:
    """Map Python type to a Click parameter type."""
    if get_origin(type_) is tuple:
        return get_args(type_)[0]

    if type_ in (int, float, str):
        return type_

    if isinstance(type_, type) and issubclass(type_, Enum):
        return click.Choice([e.value for e in type_])

    return str
