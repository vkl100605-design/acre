# Hugging Face Space: nginx:7860 → Streamlit (full ACRE++ demo) + OpenEnv FastAPI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

COPY space/nginx.conf /etc/nginx/nginx.conf
COPY . .
RUN chmod +x /app/space/entrypoint.sh

EXPOSE 7860

CMD ["/bin/bash", "/app/space/entrypoint.sh"]
