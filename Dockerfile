# syntax=docker/dockerfile:1.7
#
# Single-purpose image: serve the results dashboard. It installs BASE deps only
# (no `experiments` extra), so there is no torch/chromadb/guideline-gpt and the
# image stays small. The dashboard reads pre-computed results from the repo /
# mounted volume — no API keys required (spec §12).

# ---------- builder ----------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Base deps only — the dashboard needs nothing from the experiments extra.
RUN uv venv /app/.venv \
 && uv pip install --python /app/.venv/bin/python -e .

# ---------- runtime ----------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG UID=10001
RUN useradd --uid ${UID} --create-home --shell /bin/bash app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml README.md METHODOLOGY.md DATASET.md ./
COPY --chown=app:app src ./src
# Pre-computed results + dataset + experiment configs so the app is fully
# browseable out of the box; mountable in compose to override.
COPY --chown=app:app results ./results
COPY --chown=app:app eval_dataset ./eval_dataset
COPY --chown=app:app experiments ./experiments

USER app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)"

CMD ["streamlit", "run", "src/eval_harness/ui/dashboard.py"]
