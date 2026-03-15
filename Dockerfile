FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for common Python wheels (pdf/crypto/ML stacks).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-all.txt

COPY . .

EXPOSE 9000

CMD ["streamlit", "run", "app.py", "--server.port", "9000", "--server.address", "0.0.0.0"]
