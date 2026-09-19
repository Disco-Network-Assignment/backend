FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY app ./app
COPY prompts ./prompts
COPY data ./data
COPY alembic.ini ./
COPY migrations ./migrations
RUN pip install .

EXPOSE 8000
# bring the schema up to date, then serve
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
