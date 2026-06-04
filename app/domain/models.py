from dataclasses import dataclass


@dataclass(frozen=True)
class AppLimits:
    max_documents: int
    max_file_size_bytes: int
    max_total_size_bytes: int
    max_pdf_pages: int
    retrieval_top_k: int
    embedding_model_id: str
    upload_folder: str

