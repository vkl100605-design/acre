# Default image: lean OpenEnv API (suited for Hugging Face Spaces, port 7860).
# For full local stack (Streamlit, torch, trl) use: pip install -r requirements.txt on host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

COPY . .

# HF Spaces (Docker) expect the app on 7860; local docker can map -p 8000:7860 if needed
EXPOSE 7860

CMD ["uvicorn", "openenv_server:app", "--host", "0.0.0.0", "--port", "7860"]
