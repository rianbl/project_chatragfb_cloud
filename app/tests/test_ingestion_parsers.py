import pathlib
import sys
import unittest
from unittest.mock import mock_open, patch

APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules.ingestion_parsers import FileParserRegistry, TxtParser, build_default_parser_registry, extract_units


class _CustomParser:
    def __init__(self, label):
        self.label = label

    def extract_units(self, file_path):
        return [{"text": f"custom:{self.label}:{file_path}", "metadata": {"source_type": "custom"}}]


class IngestionParsersTests(unittest.TestCase):
    def test_default_registry_has_csv_txt_pdf(self):
        registry = build_default_parser_registry()

        self.assertIsNotNone(registry.get(".csv"))
        self.assertIsNotNone(registry.get(".txt"))
        self.assertIsNotNone(registry.get(".pdf"))

    def test_registry_accepts_custom_parser_without_core_changes(self):
        registry = FileParserRegistry(_parsers={})
        registry.register(".md", _CustomParser("markdown"))
        sample = "dummy.md"

        units = extract_units(sample, ".md", registry=registry)

        self.assertEqual(len(units), 1)
        self.assertIn("custom:markdown", units[0]["text"])

    def test_txt_parser_splits_blocks(self):
        parser = TxtParser()
        txt_file = "sample.txt"

        with patch("builtins.open", mock_open(read_data="Bloco A\n\nBloco B")):
            units = parser.extract_units(txt_file)

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["metadata"]["source_type"], "txt_block")
        self.assertEqual(units[0]["metadata"]["block_number"], 1)

    def test_csv_parser_builds_row_units_with_header(self):
        registry = build_default_parser_registry()
        csv_file = "data.csv"

        with patch("builtins.open", mock_open(read_data="nome,idade\nana,30\nbia,28\n")):
            units = extract_units(csv_file, ".csv", registry=registry)

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["metadata"]["source_type"], "csv_row")
        self.assertIn("CSV header: nome | idade", units[0]["text"])
        self.assertIn("CSV row 1: ana | 30", units[0]["text"])

    def test_unsupported_extension_raises_value_error(self):
        registry = build_default_parser_registry()
        unknown = "file.xyz"

        with self.assertRaises(ValueError):
            extract_units(unknown, ".xyz", registry=registry)


if __name__ == "__main__":
    unittest.main()
