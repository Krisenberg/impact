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
sync: _uv
    uv sync

# Recursively remove cache and .pyc/.pyo files
cleanup:
    @find . -regextype posix-extended -regex '.*(__pycache__|\.py[co])$$' -delete
    @rm -rf .venv
    @echo "Removed .venv and all __pycache__ folders and .pyc/.pyo files (recursively)"
