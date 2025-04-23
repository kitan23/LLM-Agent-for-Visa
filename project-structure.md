# OPT-RAG Project Directory Structure

```
OPT-RAG/
│── docker-compose.yml            # Container orchestration for local development
│── .env                          # Environment variables
│── .gitignore                    # Git ignore file
│── Jenkinsfile                   # CI/CD pipeline configuration
│── README.md                     # Project documentation
│── pytest.ini                    # Python test configuration
│
├─ rag-pipeline/                  # Backend (FastAPI)
│ ├─── src/                       # Source code
│ │ ├─── main.py                  # FastAPI application entrypoint
│ │ ├─── embeddings/              # Embedding generation modules
│ │ ├─── llm/                     # LLM integration modules
│ │ ├─── document_processor/      # PDF and document processing
│ │ ├─── retriever/               # Vector search and context retrieval
│ │ └─── utils/                   # Utility functions
│ │
│ ├─── vector_store/              # FAISS vector database storage
│ ├─── models/                    # Local model storage
│ ├─── examples/                  # Example documents and PDFs
│ │ ├─── opt_guide.pdf            # OPT application guide
│ │ ├─── visa_faq.pdf             # Common visa questions
│ │ └─── work_authorization.pdf   # Work authorization guidelines
│ │
│ ├─── notebooks/                 # Jupyter notebooks for prototyping
│ │ ├─── rag_pipeline_poc.ipynb   # RAG pipeline proof of concept
│ │ └─── data_preprocessing.ipynb # Document preprocessing exploration
│ │
│ ├─── tests/                     # Test suite for backend
│ ├─── uploaded_pdfs/             # Storage for user-uploaded documents
│ ├─── Dockerfile                 # Backend container definition
│ └─── requirements.txt           # Python dependencies
│
├─ streamlit/                     # Frontend (Streamlit)
│ ├─── main.py                    # Streamlit UI application
│ ├─── pages/                     # Additional UI pages
│ ├─── components/                # UI components
│ ├─── static/                    # Static assets (images, css)
│ ├─── Dockerfile                 # Frontend container definition
│ └─── requirements.txt           # Frontend dependencies
│
├─ nginx/                         # API Gateway
│ ├─── conf/                      # NGINX configuration
│ │ └─── nginx.conf               # Main configuration file
│ └─── Dockerfile                 # NGINX container definition
│
├─ grafana/                       # Monitoring dashboards
│ ├─── dashboards/                # Dashboard definitions
│ │ ├─── opt_rag_overview.json    # System overview dashboard
│ │ ├─── llm_metrics.json         # LLM performance metrics
│ │ └─── user_metrics.json        # User interaction metrics
│ ├─── provisioning/              # Automated provisioning
│ │ ├─── dashboards/              # Dashboard configurations
│ │ └─── datasources/             # Data source configurations
│ └─── Dockerfile                 # Grafana container definition
│
├─ prometheus/                    # Metrics collection
│ ├─── prometheus.yml             # Prometheus configuration
│ └─── Dockerfile                 # Prometheus container definition
│
├─ jaeger/                        # Distributed tracing
│ └─── Dockerfile                 # Jaeger container definition
│
├─ jenkins/                       # CI/CD pipeline
│ ├─── Dockerfile                 # Jenkins container definition
│ └─── jenkins-config/            # Jenkins configuration files
│
└─ kubernetes/                    # Kubernetes deployment
  ├─── local/                     # Local cluster configuration
  │ ├─── backend.yaml             # Backend service and deployment
  │ ├─── frontend.yaml            # Frontend service and deployment
  │ ├─── monitoring.yaml          # Monitoring stack
  │ └─── ingress.yaml             # Ingress configuration
  │
  └─── cloud/                     # Cloud deployment (GKE)
    ├─── backend.yaml             # Backend service and deployment
    ├─── frontend.yaml            # Frontend service and deployment
    ├─── monitoring.yaml          # Monitoring stack
    └─── ingress.yaml             # Ingress configuration
``` 