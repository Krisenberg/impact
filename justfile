# Install uv if not present
_uv:
    @command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Lock dependencies
lock: _uv
    uv lock

# Install dependencies (pass extra flags via ARGS, e.g. just install --no-dev)
install *ARGS: _uv
    uv sync --frozen {{ARGS}}

# Synchronize dependencies
sync *ARGS: _uv
    uv sync {{ARGS}}

# Check code style and formatting (no fixes applied)
lint: _uv
    uv run ruff check --no-fix
    uv run ruff format --diff

# Run static type checkers
typecheck: _uv
    uv run mypy --config-file pyproject.toml
    uv run basedpyright --stats

# Run the IMPACT CLI
impact *ARGS: _uv
    uv run cli {{ARGS}}

# Run the scripts/plots.py CLI
thesis-plots *ARGS: _uv
    uv run scripts/plots.py {{ARGS}}

# Run the scripts/tables.py CLI
thesis-tables *ARGS: _uv
    uv run scripts/tables.py {{ARGS}}

table name *ARGS: _uv
    uv run python -m scripts.cli tables {{name}} {{ARGS}}

# Recursively remove cache and .pyc/.pyo files
cleanup:
    @find . -regextype posix-extended -regex '.*(__pycache__|\.py[co])$$' -delete
    @rm -rf .venv
    @echo "Removed .venv and all __pycache__ folders and .pyc/.pyo files (recursively)"
