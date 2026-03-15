FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements*.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements-all.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN useradd --uid 1000 --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port", "9000", "--server.address", "0.0.0.0"]
