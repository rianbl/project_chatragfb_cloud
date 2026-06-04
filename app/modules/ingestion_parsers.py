from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Protocol


class FileParser(Protocol):
    def extract_units(self, file_path: str) -> list[dict]:
        ...


def _read_text_with_fallback(file_path: str) -> str:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as source:
                return source.read()
        except UnicodeDecodeError:
            continue

    with open(file_path, "r", encoding="utf-8", errors="replace") as source:
        return source.read()


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class CsvParser:
    def extract_units(self, file_path: str) -> list[dict]:
        units = []
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as source:
            reader = csv.reader(source)
            rows = [row for row in reader if any(cell.strip() for cell in row)]

        if not rows:
            return units

        header_text = " | ".join(cell.strip() for cell in rows[0] if cell.strip())
        for row_idx, row in enumerate(rows[1:], start=1):
            row_text = " | ".join(cell.strip() for cell in row if cell.strip())
            if not row_text:
                continue
            units.append(
                {
                    "text": f"CSV header: {header_text}\nCSV row {row_idx}: {row_text}",
                    "metadata": {"source_type": "csv_row", "row_number": row_idx},
                }
            )

        if not units and header_text:
            units.append(
                {
                    "text": f"CSV header: {header_text}",
                    "metadata": {"source_type": "csv_header"},
                }
            )
        return units


class TxtParser:
    def extract_units(self, file_path: str) -> list[dict]:
        text = _read_text_with_fallback(file_path)
        blocks = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]

        if blocks:
            return [
                {
                    "text": block,
                    "metadata": {"source_type": "txt_block", "block_number": idx},
                }
                for idx, block in enumerate(blocks, start=1)
            ]

        normalized = _normalize_text(text)
        if normalized:
            return [{"text": normalized, "metadata": {"source_type": "txt_raw"}}]
        return []


class PdfParser:
    def extract_units(self, file_path: str) -> list[dict]:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        units = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = _normalize_text(page.extract_text() or "")
            if not page_text:
                continue
            units.append(
                {
                    "text": page_text,
                    "metadata": {"source_type": "pdf_page", "page_number": page_number},
                }
            )
        return units


@dataclass
class FileParserRegistry:
    _parsers: dict[str, FileParser]

    def register(self, extension: str, parser: FileParser) -> None:
        normalized_ext = (extension or "").strip().lower()
        if not normalized_ext.startswith("."):
            normalized_ext = f".{normalized_ext}"
        self._parsers[normalized_ext] = parser

    def get(self, extension: str) -> FileParser:
        normalized_ext = (extension or "").strip().lower()
        if not normalized_ext.startswith("."):
            normalized_ext = f".{normalized_ext}"

        parser = self._parsers.get(normalized_ext)
        if parser is None:
            raise ValueError(f"Unsupported file format: {normalized_ext}")
        return parser


def build_default_parser_registry() -> FileParserRegistry:
    registry = FileParserRegistry(_parsers={})
    registry.register(".csv", CsvParser())
    registry.register(".txt", TxtParser())
    registry.register(".pdf", PdfParser())
    return registry


def extract_units(file_path: str, extension: str, registry: FileParserRegistry) -> list[dict]:
    parser = registry.get(extension)
    return parser.extract_units(file_path)
