FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Make src-layout imports work without installing the root project
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==2.2.1"

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi --no-root \
  && rm -rf /root/.cache/pypoetry /root/.cache/pip

COPY . /app

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
