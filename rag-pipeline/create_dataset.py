from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load the dataset
DATASET_PATH = "data/processed/documents.parquet"  # Input: existing parquet with text
original_df = pd.read_parquet(DATASET_PATH)
print(f"Original dataframe columns: {original_df.columns.tolist()}")
print(f"Original dataframe shape: {original_df.shape}")

# Create documents with metadata preserved
docs_with_metadata = []
for idx, row in original_df.iterrows():
    doc = Document(
        page_content=row['passage'],
        metadata={
            'original_chunk_id': row['chunk_id'],
            'source': row['source'],
            'file_path': row['file_path'] if 'file_path' in row else '',
            'file_size': row['file_size'] if 'file_size' in row else 0,
            'chunk_length': row['chunk_length'] if 'chunk_length' in row else len(row['passage']),
        }
    )
    docs_with_metadata.append(doc)

# Split documents while preserving metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
splits = splitter.split_documents(docs_with_metadata)

# Create new dataframe preserving all metadata
data_rows = []
for new_chunk_id, doc in enumerate(splits):
    data_rows.append({
        'chunk_id': new_chunk_id,  # New sequential chunk ID
        'passage': doc.page_content,
        'source': doc.metadata.get('source', ''),
        'file_path': doc.metadata.get('file_path', ''),
        'file_size': doc.metadata.get('file_size', 0),
        'chunk_length': len(doc.page_content),
        'original_chunk_id': doc.metadata.get('original_chunk_id', -1),
    })

df = pd.DataFrame(data_rows)
print(f"New dataframe columns: {df.columns.tolist()}")
print(f"New dataframe shape: {df.shape}")

# 3. Generate embeddings for the 'passage' column
print("Generating embeddings...")
df['embedding'] = df['passage'].apply(lambda x: model.encode(x).tolist())
df['event_timestamp'] = pd.Timestamp.now(tz='UTC')

# 4. Save the DataFrame with embeddings to a new Parquet file
OUTPUT_PATH = "data/processed/documents_with_embeddings.parquet"  # Output: parquet with embeddings

# Ensure output directory exists
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved embeddings for {len(df)} text chunks to {OUTPUT_PATH}")
