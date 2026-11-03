FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt .
EXPOSE 8000
ENV MODEL_PATH=/models/openai-gsm8k
CMD ["uvicorn", "math_modal.api:app", "--host", "0.0.0.0", "--port", "8000"]
