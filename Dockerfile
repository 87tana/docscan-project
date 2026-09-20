FROM python:3.11-slim

WORKDIR /app

# Install API dependencies first (better Docker layer caching)
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt


# Copy the code and model files the API needs
COPY api/ api/
COPY mlflow_local/ mlflow_local/


# Bridge the absolute host path baked into mlruns.db to where files actually live in the container
RUN mkdir -p /home/tannaz/Documents/Projects/Docscan_project && \
    ln -s /app/mlflow_local /home/tannaz/Documents/Projects/Docscan_project/mlflow_local


# Tell the app where "PROJECT_ROOT" is INSIDE the container
ENV PROJECT_ROOT=/app


EXPOSE 8000


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


