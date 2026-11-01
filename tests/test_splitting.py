import unittest
from pathlib import Path
from src.ingestion.splitting import get_document_splits


class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "src" / "data"

    def test_document_splitting(self):
        """Test PDF loading and chunk splitting"""
        self.assertTrue(self.data_dir.exists(), f"Data directory not found: {self.data_dir}")

        splits = get_document_splits(data_dir=str(self.data_dir), chunk_size=750, chunk_overlap=100)

        self.assertGreater(len(splits), 0, "No chunks were extracted!")
        first_chunk = splits[0]
        self.assertTrue(hasattr(first_chunk, "page_content"))
        self.assertGreater(len(first_chunk.page_content.strip()), 0, "First chunk is empty!")
        self.assertIn("source", first_chunk.metadata, "Metadata missing source key!")

        print(f"\n[PASS] Splitting test passed. Total chunks extracted: {len(splits)}")


if __name__ == "__main__":
    unittest.main()
