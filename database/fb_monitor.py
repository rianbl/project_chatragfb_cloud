from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Endpoint to receive feedback data.
    Logs the data and returns a success response.
    """
    data = request.get_json()
    
    # Log the received data
    logger.info("Received feedback data: %s", data)
    
    # Respond with a success message
    return jsonify({"message": "Data received successfully"}), 200

if __name__ == '__main__':
    # Run on port 5002 for isolation from other services
    app.run(host='0.0.0.0', port=5002)
