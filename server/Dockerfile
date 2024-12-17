FROM python:3.10

ENV POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-root --no-dev

COPY . /app
COPY .env /app/.env
COPY alembic.ini /app/alembic.ini

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
