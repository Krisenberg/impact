"""Module for configuring application-wide structured logging."""

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    json_log_file: str | Path | None = None,
    details: bool = False,
) -> None:
    """Configure application-wide logging using structlog.

    This function should be called once at the application's entry point.

    The configuration is inspired by the structlog docs:
    https://www.structlog.org/en/stable/performance.html

    Args:
        level: The minimum log level to display.
        log_file: Path to a log file with logs in a human-readable format (.log).
            If None (default), no such file will be created.
        json_log_file: Path to a log file with logs in a machine-readable format (JSON).
            If None (default), no such file will be created.
        details: If set to True, details such as a filename, line number etc. will be
            added to the log messages.
    """
    shared_processors = [
        # Add timestamps
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add log level
        structlog.processors.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add exception info on error
        structlog.dev.set_exc_info,
    ]

    if details:
        shared_processors.append(
            # Add call site information
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            )
        )

    structlog_processors = shared_processors + [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=structlog_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    # Configure the standard library logging
    handlers = [_configure_console_handler(shared_processors)]

    if log_file:
        handlers.append(
            _configure_file_handler(shared_processors, log_file, json_format=False)
        )
    if json_log_file:
        handlers.append(
            _configure_file_handler(shared_processors, json_log_file, json_format=True)
        )

    # Configure the root logger with handlers defined above
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def _configure_console_handler(
    pre_chain_processors: list[structlog.types.Processor],
) -> logging.Handler:
    """Configure the console handler."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within structlog
        foreign_pre_chain=pre_chain_processors,
        # These run on ALL entries after the pre_chain is done
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # Render the final output with nice formatting for the console
            structlog.dev.ConsoleRenderer(),
        ],
    )
    console_handler.setFormatter(console_formatter)
    return console_handler


def _configure_file_handler(
    pre_chain_processors: list[structlog.types.Processor],
    log_file: Path,
    json_format: bool = False,
) -> logging.Handler:
    """Configure the file handler."""
    file_handler = logging.FileHandler(log_file)

    if json_format:
        processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # Format any exceptions
            structlog.processors.format_exc_info,
            # Rename the 'event' (default) key in event dicts to 'msg'
            structlog.processors.EventRenamer(to="msg"),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Use the same ConsoleRenderer as for the console output but without colors
        processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # Format any exceptions
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False),
        ]

    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain_processors,
        processors=processors,
    )
    file_handler.setFormatter(file_formatter)
    return file_handler
