"""
Unified Entry Point for Academic Research RAG.
Defaults to the Retrieval / Query Pipeline, while also supporting the Ingestion command.
"""
import sys
from query import main as query_main
from ingest import main as ingest_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        sys.argv.pop(1)
        ingest_main()
    else:
        query_main()
