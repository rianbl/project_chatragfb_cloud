FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY chat/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy chat script
COPY chat/chat_llm.py /app/

# Expose port for the chat application
EXPOSE 8081

# Run the chat application
CMD ["python", "/app/chat_llm.py"]
