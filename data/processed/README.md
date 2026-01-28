# Processed Data Directory

This directory contains processed/transformed data files.

## Structure

- `chunks/`: Document chunks after processing
- `embeddings/`: Serialized embeddings  
- `index/`: Vector index files
- `metadata/`: Processing metadata and logs

## Processing Pipeline

1. Raw documents in `../raw/` are loaded
2. Documents are split into chunks
3. Chunks are embedded using selected model
4. Embeddings are stored in vector database
5. Metadata is saved for tracking

## Notes

- Never commit processed data to version control
- Use `.gitignore` to exclude this directory
- Processed data should be regenerated from raw sources