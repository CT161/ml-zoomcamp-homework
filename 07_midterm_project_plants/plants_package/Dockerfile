# Use the official lightweight Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory inside the container
WORKDIR /app

# Enable bytecode compilation and copy mode for cache mounts
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

# Step 1: Install dependencies first using pyproject.toml and uv.lock
# (Ensure you run `uv lock` in plants_package/ first if you haven't yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project

# Step 2: Copy the rest of the project source code and files into the container
ADD . /app

# Step 3: Run sync again to install your package into the environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Place the virtual environment's binaries at the front of the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for FastAPI / web service
EXPOSE 8000

# Default command to run your prediction module or script inside the virtual environment
CMD ["uv", "run", "python", "src/plants_app/predict.py"]

