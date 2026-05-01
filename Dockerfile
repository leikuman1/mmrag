FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs
COPY examples ./examples
COPY APPLICATION_ANSWER_zh.md ./

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "mmrag.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]

