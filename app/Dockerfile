FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /app/requirements.txt

COPY app /app

EXPOSE 8080

CMD ["python", "server.py"]
