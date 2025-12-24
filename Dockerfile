FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libjpeg-dev \
    zlib1g-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN useradd --create-home --shell /bin/false appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
