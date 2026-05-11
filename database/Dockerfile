FROM postgres:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=admin
ENV POSTGRES_DB=llm_data

# Set working directory for init scripts
WORKDIR /

# Copy initialization scripts and data
COPY /database /database

# Install Python and setup virtual environment
COPY database/requirements.txt .
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Add the virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Expose PostgreSQL and data loader API ports
EXPOSE 5432 5001 5002

# Copy data loader and schema scripts
COPY database/data_loader.py /data_loader.py
COPY database/schema_builder.py /schema_builder.py
COPY database/populate.py /populate.py
COPY database/fb_monitor.py /fb_monitor.py

# Entrypoint: source secrets and start services
CMD ["bash", "-c", "export POSTGRES_PASSWORD=$(cat /run/secrets/postgres_password) && docker-entrypoint.sh postgres & python3 /data_loader.py & python3 /fb_monitor.py"]
