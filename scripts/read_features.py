from feast import FeatureStore
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

model = SentenceTransformer("all-MiniLM-L6-v2")
store = FeatureStore("./feature_store")
# query = "What do beetles eat?"
query = "What is STEM OPT?"

embedding = model.encode(query).tolist()
print("Query embedding:", len(embedding))

context_data = store.retrieve_online_documents_v2(
    features=[
        "document_embeddings:embedding",
       "document_embeddings:chunk_id", 
        "document_embeddings:passage",
        "document_embeddings:source",
    ],
    query=embedding,
    top_k=3,
    distance_metric="COSINE",
).to_df()

print(f"Found {len(context_data)} results:")
print(f"Available columns: {context_data.columns.tolist()}")
print()

# Print each result in a readable format with similarity score
for i, row in context_data.iterrows():
    print(f"=== Result {i+1} ===")
    print(f"Source: {row['source']}")
    print(f"Chunk ID: {row['chunk_id']}")
    
    # Check if there's a distance/score column
    if 'distance' in row:
        print(f"Distance: {row['distance']:.4f}")
    elif 'score' in row:
        print(f"Score: {row['score']:.4f}")
    elif 'similarity' in row:
        print(f"Similarity: {row['similarity']:.4f}")
    
    print(f"Content: {row['passage']}")
    print("-" * 80)