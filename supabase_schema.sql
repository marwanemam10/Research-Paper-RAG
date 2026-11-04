-- ================================================================
-- Supabase Schema for Academic Research Paper RAG Pipeline
-- ================================================================

-- 1. Create Storage Bucket for Raw PDF Papers (if not already existing)
INSERT INTO storage.buckets (id, name, public)
VALUES ('research-papers', 'research-papers', false)
ON CONFLICT (id) DO NOTHING;

-- 2. Create Document Status Enum & Documents Table
DO $$ BEGIN
    CREATE TYPE public.document_status AS ENUM ('pending', 'processing', 'indexed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS public.documents (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    file_size BIGINT NOT NULL DEFAULT 0,
    storage_path TEXT NOT NULL,
    status public.document_status NOT NULL DEFAULT 'pending',
    total_chunks INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Indexes for Fast Lookup & Deduplication
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON public.documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents(status);

-- 4. Automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trigger_documents_updated_at ON public.documents;
CREATE TRIGGER trigger_documents_updated_at
    BEFORE UPDATE ON public.documents
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

