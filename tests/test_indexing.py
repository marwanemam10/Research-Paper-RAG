import unittest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from src.ingestion.indexing import get_qdrant_client, index_documents, get_vector_store


class TestIndexing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection_name = "test_rag_collection"
        cls.client = get_qdrant_client()
        # FakeEmbeddings allows fast testing without calling external API
        cls.fake_embeddings = FakeEmbeddings(size=768)

    def test_1_client_connection(self):
        """Test connection to Qdrant server"""
        try:
            collections = self.client.get_collections()
            self.assertIsNotNone(collections)
            print("\n[PASS] Qdrant connection verified successfully.")
        except Exception as e:
            self.fail(f"Failed to connect to Qdrant: {e}")

    def test_2_index_and_search_documents(self):
        """Test indexing dummy documents and similarity search"""
        dummy_docs = [
            Document(page_content="Attention is all you need introduces Transformers.", metadata={"source": "test1"}),
            Document(page_content="Qdrant is a high performance vector database.", metadata={"source": "test2"})
        ]

        vector_store = index_documents(
            documents=dummy_docs,
            collection_name=self.collection_name,
            client=self.client,
            embedding=self.fake_embeddings,
            recreate_collection=True
        )

        self.assertIsNotNone(vector_store, "Vector store should not be None.")

        # Similarity search test
        results = vector_store.similarity_search("Transformers", k=1)
        self.assertGreater(len(results), 0, "Search returned no results.")
        self.assertIsInstance(results[0], Document)
        print(f"[PASS] Search returned matching document: '{results[0].page_content}'")

    def test_3_get_vector_store(self):
        """Test retrieving vector store from an existing collection"""
        store = get_vector_store(
            collection_name=self.collection_name,
            client=self.client,
            embedding=self.fake_embeddings
        )
        self.assertIsNotNone(store)
        print("[PASS] Successfully retrieved existing vector store instance.")

    @classmethod
    def tearDownClass(cls):
        # Clean up test collection after all tests complete
        if cls.client.collection_exists(cls.collection_name):
            cls.client.delete_collection(cls.collection_name)
            print(f"[CLEANUP] Deleted test collection: '{cls.collection_name}'.")


if __name__ == "__main__":
    unittest.main()
