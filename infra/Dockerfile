FROM apache/airflow:2.9.0

USER airflow

RUN pip install uv --upgrade

COPY feast-0.1.dev4056+g90e627f.d20250627-py2.py3-none-any.whl .

RUN uv pip install \
    "apache-airflow==2.9.0" \
    "apache-airflow-providers-docker" \
    "boto3" \
    "click" \
    "elasticsearch" \
    "fastapi" \
    "feast @ feast-0.1.dev4056+g90e627f.d20250627-py2.py3-none-any.whl" \
    "fsspec" \
    "huggingface-hub" \
    "ipython" \
    "langchain" \
    "langchain-community" \
    "langchain-openai" \
    "langgraph" \
    "minio" \
    "numpy>=1.22.4" \
    "pandas" \
    "protobuf>=6.0.0" \
    "psycopg2-binary" \
    "pydantic" \
    "pymilvus" \
    "pydantic-settings" \
    "sentence-transformers" 