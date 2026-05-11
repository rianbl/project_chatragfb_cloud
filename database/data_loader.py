from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import requests  # Import requests library
from schema_builder import create_table  # Importing the create_table function
from populate import populate_table
import logging

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        app.logger.error("No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No file selected.")
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if the directory is empty
    if os.listdir(UPLOAD_FOLDER):
        # Directory is not empty, replace the existing file
        for existing_file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, existing_file))
        app.logger.info("File already exists. The existing file has been replaced.")
        message = "File already exists. The existing file has been replaced."
    else:
        app.logger.info("File uploaded successfully.")
        message = "File uploaded successfully."

    # Save the new file
    file.save(filepath)

    # STEP 1 BUILD SCHEMA
    try:
        create_table(filepath)  # Pass the file path to create the table
        app.logger.info("Schema created successfully.")
        message += " Schema created successfully."
    except Exception as e:
        app.logger.error(f"Error creating schema: {str(e)}")
        message += f" Error creating schema: {str(e)}"
    
    # STEP 2 POPULATE TABLE
    try:
        populate_table()  # Pass the file path to create the table
        app.logger.info("Table populated successfully.")
        message += " Table populated successfully."
    except Exception as e:
        app.logger.error(f"Error populating table: {str(e)}")
        message += f" Error populating table: {str(e)}"

    # STEP 3 BUILD VECTOR DATABASE
    try:
        response = requests.post(f"{SEARCH_ENGINE_API_URL}/refresh")
        if response.status_code == 200:
            app.logger.info("Vector database built successfully.")
            message += " Vector database built successfully."
        else:
            app.logger.error(f"Error building vector database: {response.json().get('error', 'Unknown error')}.")
            message += f" Error building vector database: {response.json().get('error', 'Unknown error')}."
    except Exception as e:
        app.logger.error(f"Error calling search engine API: {str(e)}")
        message += f" Error calling search engine API: {str(e)}"

    return jsonify({"message": message, "file_path": filepath}), 200

@app.route('/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve log messages."""
    return jsonify({"logs": log_messages}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
