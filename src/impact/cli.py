import dataclasses
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import click

from impact.dataset import DatasetSize, DatasetType, InteractionType
from impact.pipeline import PipelineConfig

if TYPE_CHECKING:
    from enum import StrEnum


def inject_pipeline_config(f: Callable[..., Any]) -> Callable[..., Any]:
    """Inject PipelineConfig fields as Click options and pass the instance as `config`.

    The decorated function must accept `config: PipelineConfig` as a keyword argument.
    """
    # Define all click options, matching the defaults from PipelineConfig
    options = [
        click.option(
            "--dataset-size",
            type=click.Choice([e.value for e in DatasetSize]),
            required=True,
            help="Yambda dataset variant to process (50M, 500M, or 5B)",
        ),
        click.option(
            "--dataset-type",
            type=click.Choice([e.value for e in DatasetType]),
            default=DatasetType.FLAT.value,
            show_default=True,
            help="Dataset format type. Currently only FLAT is supported.",
        ),
        click.option(
            "--interaction-type",
            type=click.Choice([e.value for e in InteractionType]),
            default=InteractionType.MULTI_EVENT.value,
            show_default=True,
            help="""Interaction file to load.

            MULTI_EVENT includes all event types (listen, like, etc.); other variants
            contain only a subset.
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
            help="""Number of tracks that consitute a session. Denoted as K in thesis.

            Sessions with less than K tracks are discarded. Sessions with more than K
            tracks are truncated.
            """,
        ),
        click.option(
            "--min-listen-seconds",
            type=int,
            default=30,
            show_default=True,
            help="""
            Minimum playback duration (in seconds) to consider an event as a valid
            listen.

            Defaults to 30 (industry standard used by authors of other publications).
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
            help="Step size for the sliding window sequence generation.",
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
                Buffer time (in minutes) between dataset splits.

                The purpose of this parameter is to mimic production system deployment
                latency.
            """,
        ),
    ]

    # Apply Click options to the function in rev order (to maintain defined ordering)
    for option in reversed(options):
        f = option(f)

    @wraps(f)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract fields belonging to PipelineConfig
        config_kwargs: dict[str, Any] = {}
        enum_by_field_name: dict[str, type[StrEnum]] = {
            "dataset_size": DatasetSize,
            "dataset_type": DatasetType,
            "interaction_type": InteractionType,
        }
        for field in dataclasses.fields(PipelineConfig):
            if field.name not in kwargs:
                continue
            if field.name in enum_by_field_name:
                # Parse string choice back into its respective Enum class
                enum = enum_by_field_name[field.name]
                config_kwargs[field.name] = enum(kwargs.pop(field.name))
            else:
                config_kwargs[field.name] = kwargs.pop(field.name)

        config = PipelineConfig(**config_kwargs)

        # Inject the config object and forward remaining CLI options
        return f(*args, config=config, **kwargs)

    return _wrapper
