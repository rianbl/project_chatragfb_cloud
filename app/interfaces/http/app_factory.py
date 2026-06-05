from __future__ import annotations

import logging
import os

from bootstrap.container import build_container
from domain.models import AppLimits
from flask import Flask
from flask_cors import CORS
from modules.config import (
    EMBEDDING_MODEL_ID,
    MAX_DOCUMENTS,
    MAX_FILE_SIZE_BYTES,
    MAX_PDF_PAGES,
    MAX_TOTAL_SIZE_BYTES,
    RETRIEVAL_TOP_K,
    UPLOAD_FOLDER,
)

from .routes import register_routes


class ListHandler(logging.Handler):
    def __init__(self, collector: list[str], formatter: logging.Formatter) -> None:
        super().__init__()
        self._collector = collector
        self.setFormatter(formatter)

    def emit(self, record):
        self._collector.append(self.format(record))


def create_app() -> Flask:
    app = Flask(__name__, static_folder="../../static", static_url_path="")
    CORS(app)

    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure root logger to capture logs from all modules
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_handler = logging.StreamHandler()
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_handler)

    log_messages: list[str] = []
    root_logger.addHandler(ListHandler(log_messages, log_formatter))

    # Ensure app.logger also uses the same level
    app.logger.setLevel(logging.INFO)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    limits = AppLimits(
        max_documents=MAX_DOCUMENTS,
        max_file_size_bytes=MAX_FILE_SIZE_BYTES,
        max_total_size_bytes=MAX_TOTAL_SIZE_BYTES,
        max_pdf_pages=MAX_PDF_PAGES,
        retrieval_top_k=RETRIEVAL_TOP_K,
        embedding_model_id=EMBEDDING_MODEL_ID,
        upload_folder=UPLOAD_FOLDER,
    )
    container = build_container(limits=limits, logger=app.logger)
    app.config["service_container"] = container

    register_routes(app, container, log_messages)
    return app
