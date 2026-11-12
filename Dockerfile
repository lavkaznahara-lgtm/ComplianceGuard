# syntax=docker/dockerfile:1
# ComplianceGuard: Streamlit dashboard + batch CLI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    CG_EVALUATION__CHECKPOINT_PATH=/app/runs/results_partial.jsonl \
    CG_EVALUATION__OUTPUT_PATH=/app/runs/results.jsonl

# Unprivileged user; /app/runs is the only writable project folder.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/runs /app/.tiktoken \
    && chown -R appuser:appuser /app
WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code only (no tests, no local result files).
COPY --chown=appuser:appuser config.yaml config.py api_client.py validator.py main.py app.py ./
COPY --chown=appuser:appuser core ./core
COPY --chown=appuser:appuser data ./data

USER appuser

# Optional warm-up so token counting is exact offline later; the build still
# succeeds without network (the client falls back to an estimate).
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" \
    || echo "tiktoken warm-up skipped (no network at build time)"

EXPOSE 8501
VOLUME ["/app/runs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()==b'ok' else 1)"

# The API key is never baked into the image: pass it at run time
# (docker run -e OPENAI_API_KEY ...). Without it the app starts in Demo Mode.
CMD ["streamlit", "run", "app.py"]