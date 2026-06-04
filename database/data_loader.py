from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import requests  # Import requests library
from schema_builder import create_schema
from populate import populate_table, SUPPORTED_EXTENSIONS
import logging
import time

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])

UPLOAD_FOLDER = 'uploads'
SEARCH_ENGINE_API_URL = "http://search:5000"  # Replace with the actual search engine host URL
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logger
app.logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(message)s')
log_handler.setFormatter(log_formatter)
app.logger.addHandler(log_handler)

# Store logs in a list
log_messages = []

class ListHandler(logging.Handler):
    def emit(self, record):
        log_messages.append(self.format(record))

list_handler = ListHandler()
list_handler.setFormatter(log_formatter)
app.logger.addHandler(list_handler)

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _is_supported_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in SUPPORTED_EXTENSIONS


def _build_file_path(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid filename.")

    base_name, extension = os.path.splitext(safe_name)
    candidate = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(candidate):
        return candidate

    timestamp = int(time.time())
    deduped = f"{base_name}_{timestamp}{extension}"
    return os.path.join(app.config['UPLOAD_FOLDER'], deduped)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        app.logger.error("No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No file selected.")
        return jsonify({"error": "No file selected"}), 400

    if not _is_supported_file(file.filename):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        app.logger.error("Unsupported file format for upload.")
        return jsonify({"error": f"Unsupported file format. Supported: {supported}"}), 400

    filepath = _build_file_path(file.filename)
    filename = os.path.basename(filepath)

    file.save(filepath)
    message_parts = [f"File '{filename}' uploaded successfully."]

    # STEP 1 BUILD SCHEMA
    try:
        create_schema()
        app.logger.info("Schema created successfully.")
        message_parts.append("Schema created successfully.")
    except Exception as e:
        app.logger.error(f"Error creating schema: {str(e)}")
        return jsonify({"error": f"Error creating schema: {str(e)}"}), 500
    
    # STEP 2 INGEST + CHUNK
    try:
        ingestion_result = populate_table(file_path=filepath, filename=filename)
        app.logger.info("File ingested and chunked successfully.")
        message_parts.append(
            f"Ingestion completed with {ingestion_result['chunks_inserted']} chunks."
        )
    except Exception as e:
        app.logger.error(f"Error ingesting file: {str(e)}")
        return jsonify({"error": f"Error ingesting file: {str(e)}"}), 500

    # STEP 3 BUILD VECTOR DATABASE
    try:
        response = requests.post(f"{SEARCH_ENGINE_API_URL}/refresh", timeout=60)
        if response.status_code == 200:
            app.logger.info("Vector database built successfully.")
            message_parts.append("Vector database built successfully.")
            status_code = 200
        else:
            search_error = response.json().get('error', 'Unknown error')
            app.logger.error(f"Error building vector database: {search_error}.")
            message_parts.append(f"Error building vector database: {search_error}.")
            status_code = 502
    except Exception as e:
        app.logger.error(f"Error calling search engine API: {str(e)}")
        message_parts.append(f"Error calling search engine API: {str(e)}")
        status_code = 502

    return jsonify(
        {
            "message": " ".join(message_parts),
            "file_path": filepath,
            "ingestion": ingestion_result,
        }
    ), status_code

@app.route('/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve log messages."""
    return jsonify({"logs": log_messages}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
