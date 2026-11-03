import unittest
from pathlib import Path
from src.storage.supabase_client import compute_file_hash, get_supabase_client


class TestSupabaseUtils(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "src" / "data"
        self.sample_pdf = self.data_dir / "1502.03167v3.pdf"

    def test_compute_file_hash(self):
        """Test SHA-256 hash calculation for deduplication"""
        self.assertTrue(self.sample_pdf.exists(), f"Sample PDF not found: {self.sample_pdf}")
        file_hash = compute_file_hash(self.sample_pdf)
        self.assertEqual(len(file_hash), 64, "SHA-256 hash should be exactly 64 characters long.")

        # Test deterministic hash (same file produces same hash)
        second_hash = compute_file_hash(self.sample_pdf)
        self.assertEqual(file_hash, second_hash, "Hash should be deterministic.")
        print(f"\n[PASS] SHA-256 hash verified: {file_hash[:16]}...")

    def test_client_init_graceful_handling(self):
        """Test that get_supabase_client handles missing credentials without crashing"""
        client = get_supabase_client()
        # If env variables are not present, it safely returns None
        if client is None:
            print("\n[INFO] Supabase credentials not set in .env (returned None as expected).")
        else:
            print("\n[PASS] Supabase client initialized with environment variables.")


if __name__ == "__main__":
    unittest.main()
