import dataclasses
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import click

from impact.config import ImpactConfig, SASRecConfig
from impact.dataset import DatasetSize, DatasetType, InteractionType

if TYPE_CHECKING:
    from enum import StrEnum


def inject_impact_config(f: Callable[..., Any]) -> Callable[..., Any]:
    """Inject ImpactConfig fields as Click options and pass the instantiated config.

    The decorated function must accept `impact_config: ImpactConfig` as a keyword
    argument.
    """
    # Define all click options, matching the defaults from ImpactConfig
    options = [
        click.option(
            "--dataset-size",
            type=click.Choice([e.value for e in DatasetSize]),
            default=DatasetSize.SIZE_50M.value,
            show_default=True,
            help="""
                Yambda dataset variant to process (50M, 500M, or 5B).
            """,
        ),
        click.option(
            "--dataset-type",
            type=click.Choice([DatasetType.FLAT.value]),
            default=DatasetType.FLAT.value,
            show_default=True,
            help="""
                Dataset format type. Currently only flat is supported.
            """,
        ),
        click.option(
            "--interaction-type",
            type=click.Choice([e.value for e in InteractionType]),
            default=InteractionType.MULTI_EVENT.value,
            show_default=True,
            help="""
                Type of interactions to load from the Yambda dataset. multi_event
                includes all event types (listen, like, etc.); other variants contain
                only a specific subset.
            """,
        ),
        click.option(
            "--inactivity-gap-seconds",
            type=int,
            default=1200,
            show_default=True,
            help="""
                Number of seconds of inactivity before a session is considered finished.
            """,
        ),
        click.option(
            "--tracks-per-session",
            type=int,
            default=10,
            show_default=True,
            help="""
                Number of tracks that consitute a session. Denoted as K in the thesis.
                Sessions with less than K tracks are discarded. Sessions with more than
                K tracks are truncated.
            """,
        ),
        click.option(
            "--min-listen-seconds",
            type=int,
            default=30,
            show_default=True,
            help="""
                Minimum playback duration (in seconds) to consider an event as a valid
                listen. Defaults to 30 (industry standard used by authors of other
                publications).
            """,
        ),
        click.option(
            "--context-sessions",
            type=int,
            default=20,
            show_default=True,
            help="""
                Number of historical sessions used as context. Denoted as L in thesis.
            """,
        ),
        click.option(
            "--step",
            type=int,
            default=5,
            show_default=True,
            help="""
                Step size for the sliding window sequence generation.
            """,
        ),
        click.option(
            "--test-days",
            type=float,
            default=1.0,
            show_default=True,
            help="""
                Number of days at the end of the timeline allocated for the test set.
            """,
        ),
        click.option(
            "--val-days",
            type=float,
            default=1.0,
            show_default=True,
            help="""
                Number of days preceding test set allocated for the validation set.
            """,
        ),
        click.option(
            "--gap-minutes",
            type=float,
            default=30.0,
            show_default=True,
            help="""
                Buffer time (in minutes) between dataset splits. The purpose of this
                parameter is to mimic production system deployment latency.
            """,
        ),
        click.option(
            "--sasrec-embedding-dim",
            type=int,
            default=64,
            show_default=True,
            help="""
                (SASRecNBR) Dimension of item and positional embeddings.
            """,
        ),
        click.option(
            "--sasrec-num-heads",
            type=int,
            default=2,
            show_default=True,
            help="""
                (SASRecNBR) Number of attention heads in each Transformer layer.
            """,
        ),
        click.option(
            "--sasrec-num-layers",
            type=int,
            default=2,
            show_default=True,
            help="""
                (SASRecNBR) Number of Transformer encoder layers.
            """,
        ),
        click.option(
            "--sasrec-dropout",
            type=float,
            default=0.2,
            show_default=True,
            help="""
                (SASRecNBR) Dropout probability applied after embedding and within each
                layer.
            """,
        ),
        click.option(
            "--sasrec-batch-size",
            type=int,
            default=2048,
            show_default=True,
            help="""
                (SASRecNBR) Number of sequences per training batch.
            """,
        ),
        click.option(
            "--sasrec-max-epochs",
            type=int,
            default=100,
            show_default=True,
            help="""
                (SASRecNBR) Maximum number of training epochs to run.
            """,
        ),
        click.option(
            "--sasrec-learning-rate",
            type=float,
            default=0.001,
            show_default=True,
            help="""
                (SASRecNBR) Learning rate for the optimizer.
            """,
        ),
        click.option(
            "--sasrec-k",
            type=int,
            default=10,
            show_default=True,
            help="""
                (SASRecNBR) Cutoff for NDCG@K and Recall@K metrics used during model
                training.
            """,
        ),
        click.option(
            "--sasrec-patience",
            type=int,
            default=10,
            show_default=True,
            help="""
                (SASRecNBR) Number of epochs to wait for validation improvement before
                stopping.
            """,
        ),
        click.option(
            "--eval-ks",
            type=tuple[int, ...],
            default=(5, 10, 40, 100),
            show_default=True,
            help="""
                Tuple of K cut-off values for computing NDCG@K and Recall@K during
                evaluation.
            """,
        ),
        click.option(
            "--seed",
            type=int,
            default=42,
            show_default=True,
            help="""
                Global random seed for strict reproducibility.
            """,
        ),
        click.option(
            "--num-workers",
            type=int,
            default=4,
            show_default=True,
            help="""
                Number of subprocesses to use for data loading.

                Set to 0 to load data in the main process, or higher for parallel
                loading (hardware-dependent).
            """,
        ),
    ]

    # Apply Click options to the function in rev order (to maintain defined ordering)
    for option in reversed(options):
        f = option(f)

    @wraps(f)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract fields belonging to ImpactConfig
        sasrec_kwargs: dict[str, Any] = {}
        impact_kwargs: dict[str, Any] = {}
        remaining_kwargs: dict[str, Any] = {}

        sasrec_field_names = {f.name for f in dataclasses.fields(SASRecConfig)}
        impact_field_names = {f.name for f in dataclasses.fields(ImpactConfig)}

        enum_by_field_name: dict[str, type[StrEnum]] = {
            "dataset_size": DatasetSize,
            "dataset_type": DatasetType,
            "interaction_type": InteractionType,
        }
        for key, value in kwargs.items():
            print(key, value)
            if key.startswith("sasrec_"):
                nested_key = key.removeprefix("sasrec_")
                if nested_key in sasrec_field_names:
                    sasrec_kwargs[nested_key] = value
            elif key in impact_field_names:
                if key in enum_by_field_name:
                    # Parse string choice back into its respective Enum class
                    enum = enum_by_field_name[key]
                    impact_kwargs[key] = enum(value)
                else:
                    impact_kwargs[key] = value
            else:
                remaining_kwargs[key] = value

        impact_kwargs["sasrec"] = SASRecConfig(**sasrec_kwargs)
        impact_config = ImpactConfig(**impact_kwargs)

        # Inject the config object and forward remaining CLI options
        return f(*args, impact_config=impact_config, **remaining_kwargs)

    return _wrapper
