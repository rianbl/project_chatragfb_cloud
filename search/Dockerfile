FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY search/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt


# Install curl
RUN apt-get update && apt-get install -y curl
# Copy the scripts and other necessary files to the container
COPY search/search_engine.py /app/search_engine.py

EXPOSE 5000
# Default command keeps the container alive for manual execution
CMD ["python", "search_engine.py"]

