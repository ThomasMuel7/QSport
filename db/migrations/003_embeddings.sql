-- À exécuter dans l'éditeur SQL Supabase ou via supabase CLI

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS match_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,           -- texte du chunk (description du match)
  metadata JSONB,                  -- infos : league, teams, date, source
  embedding vector(384),           -- embedding all-MiniLM-L6-v2
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS match_embeddings_embedding_idx
ON match_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(384),
  match_count INT DEFAULT 5
) RETURNS TABLE (
  id UUID, content TEXT, metadata JSONB, similarity FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT id, content, metadata,
    1 - (embedding <=> query_embedding) AS similarity
  FROM match_embeddings
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
