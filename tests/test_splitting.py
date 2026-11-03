import unittest
from pathlib import Path
from src.ingestion.splitting import get_document_splits, split_pdf


class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "src" / "data"
        self.sample_pdf = self.data_dir / "1502.03167v3.pdf"

    def test_single_pdf_splitting(self):
        """Test splitting a single PDF file"""
        self.assertTrue(self.sample_pdf.exists(), f"Sample PDF not found: {self.sample_pdf}")
        chunks = split_pdf(self.sample_pdf, chunk_size=750, chunk_overlap=100)
        self.assertGreater(len(chunks), 0, "No chunks extracted from single PDF!")
        first_chunk = chunks[0]
        self.assertTrue(hasattr(first_chunk, "page_content"))
        self.assertGreater(len(first_chunk.page_content.strip()), 0)
        self.assertIn("source", first_chunk.metadata)
        print(f"\n[PASS] Single PDF splitting passed. Total chunks: {len(chunks)}")

    def test_document_splitting_directory(self):
        """Test directory PDF loading and chunk splitting"""
        self.assertTrue(self.data_dir.exists(), f"Data directory not found: {self.data_dir}")

        splits = get_document_splits(data_dir=str(self.data_dir), chunk_size=750, chunk_overlap=100)

        self.assertGreater(len(splits), 0, "No chunks were extracted!")
        first_chunk = splits[0]
        self.assertTrue(hasattr(first_chunk, "page_content"))
        self.assertGreater(len(first_chunk.page_content.strip()), 0, "First chunk is empty!")
        self.assertIn("source", first_chunk.metadata, "Metadata missing source key!")

        print(f"\n[PASS] Directory splitting test passed. Total chunks extracted: {len(splits)}")


if __name__ == "__main__":
    unittest.main()
