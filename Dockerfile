FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir uv==0.11.6
RUN uv pip install --system --torch-backend cpu -r requirements.lock
COPY src/ ./src/
COPY scripts/bootstrap_demo.py ./scripts/bootstrap_demo.py
COPY data/demo/ ./data/demo/
RUN pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "medrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
