# 1. Základní image: Python 3.14 na Debianu trixie (slim = bez zbytečností).
#    debian:trixie-slim nemá Python ani pip → "pip: not found".
FROM python:3.14-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 2. Pracovní složka
WORKDIR /app

# 3. Závislosti a kód
COPY pyproject.toml .
COPY src ./src

# Instaluju aktuální složku "." (kde je pyproject.toml), v editable režimu -e:
# config.py zůstane v /app/src/docassist/config.py → parents[2] = /app → DATA_DIR = /app/data.
# Bez -e by config.py byl v site-packages a parents[2] by ukazoval do /usr/local/lib.
RUN pip install --no-cache-dir -e ".[dev]"

COPY scripts ./scripts
COPY tests ./tests

# 4. Bezpečnost: běží jako běžný uživatel, ne root.
#    /app/data v image zatím není (data připojujeme přes -v), proto ji nejdřív vytvořím.
#    chown syntaxe: chown -R <uživatel>:<skupina> <cesta>
RUN useradd --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

# 5. Síť a spuštění
EXPOSE 8000

# Docker sám volá /health
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "docassist.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
