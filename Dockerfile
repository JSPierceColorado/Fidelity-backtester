# syntax=docker/dockerfile:1.6
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Denver

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ./

# Default envs (override in Railway)
ENV INIT_CASH=10000 \
    BUY_AMOUNT=50 \
    BAR_MINUTES=15 \
    STOCKS="VIG,GLD,BND" \
    CRYPTO="BTC/USD"

# Health: quick import check
RUN python -c "import pandas, numpy, alpaca; print('imports ok')" || true

CMD ["python","/app/main.py"]
