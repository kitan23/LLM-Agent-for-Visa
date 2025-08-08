from feast import FeatureView, ValueType, Entity, FileSource, Field
from feast.types import Array, Float32, String, Int64
from feast.data_format import ParquetFormat
from datetime import timedelta

# Define the document entity
document = Entity(
    name="chunk_id",
    description="Document chunk ID for RAG pipeline",
    value_type=ValueType.INT64,
)

# Source pointing to your processed documents with embeddings
source = FileSource(
    file_format=ParquetFormat(),
    path="../rag-pipeline/data/processed/documents_with_embeddings.parquet",
    event_timestamp_column="event_timestamp",
)

# Define the feature view for document embeddings and metadata
docs_embeddings_feature_view = FeatureView(
    name="document_embeddings",
    entities=[document],
    schema=[
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,              # Enable vector search
            vector_search_metric="L2",      # Distance metric for similarity
        ),
        Field(name="passage", dtype=String),
        # Field(name="source", dtype=String),  # Commented out for now
    ],
    source=source,
    ttl=timedelta(hours=24),  # Documents stay relevant for 24 hours
)

# Note: We don't need a separate metadata feature view since 
# all metadata is already included in the document_embeddings feature view 