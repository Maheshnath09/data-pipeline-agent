# Builder stage: Install dependencies with uv
FROM python:3.12-slim AS builder
# Copy uv binary from official image (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /bin/uv
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./
# Install runtime deps only (no dev, no editable project)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Copy your app code
COPY . .

# Runtime stage: Copy venv and app for slim image
FROM python:3.12-slim AS runtime
WORKDIR /app
# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
# Copy application code from builder
COPY --from=builder /app /app
# Install system libs if needed (e.g., for matplotlib/seaborn in visualizations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set PATH for uv venv
ENV PATH="/app/.venv/bin:$PATH"
# Expose FastAPI port
EXPOSE 8000

# Run your FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]