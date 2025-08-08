# OPT-RAG: International Student Visa Assistant PRD

## Project Overview
OPT-RAG is an enterprise-grade Retrieval-Augmented Generation (RAG) system designed to help international students navigate visa-related issues, OPT applications, study/work authorization questions, and other immigration concerns. The system leverages modern MLOps practices, distributed data processing, and advanced observability to provide accurate, context-aware responses from official documentation.

## Target Users
- **Primary**: International students in the United States
- **Secondary**: University international student advisors and administrators
- **Tertiary**: Immigration support staff and prospective international students

## Problem Statement
International students face complex visa regulations, application processes, and work authorization requirements that can be difficult to navigate. Accurate information is critical but often scattered across multiple sources, difficult to interpret, or not easily accessible when needed most. Traditional search methods fail to provide contextual, personalized guidance for specific visa situations.

## Project Goals
1. **Reliable AI Assistant**: Create a production-ready AI system for international student visa queries
2. **Comprehensive Coverage**: Provide accurate information on OPT, CPT, F-1/J-1 visa status, work authorization, and related topics
3. **Operational Efficiency**: Reduce information search time and improve accuracy through intelligent document retrieval
4. **Proactive Support**: Help prevent visa status issues through timely, accurate information delivery
5. **Scalable Platform**: Build infrastructure that can expand to cover additional international student services

## System Architecture

### Core Technology Stack

**Frontend & Backend:**
- **Frontend**: Streamlit UI (Port 8501) - Interactive chat interface
- **Backend**: FastAPI service (Port 8000) - RESTful API with streaming support
- **Language**: Python 3.10 with modern async/await patterns

**LLM Infrastructure:**
- **Primary**: RunPod serverless GPU hosting for Qwen2.5-1.5B model
- **Fallback**: OpenAI GPT-4o-mini API
- **Local Models**: Qwen2.5-0.5B and Qwen2.5-1.5B for development/offline use

**Data Pipeline & Processing:**
- **Orchestration**: Apache Airflow with CeleryExecutor
- **Vector Database**: Milvus (v2.3.1) with IVF_FLAT indexing
- **Feature Store**: Feast with PostgreSQL registry and Milvus online store
- **Document Processing**: PyPDF/PyMuPDF with RecursiveCharacterTextSplitter
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)

**Infrastructure & Storage:**
- **Containerization**: Docker + Docker Compose for local development
- **Cloud Deployment**: Google Kubernetes Engine (GKE) with Helm charts
- **Object Storage**: MinIO for intermediate data and backups
- **Databases**: PostgreSQL (metadata), Redis (caching), ClickHouse (analytics)
- **Coordination**: etcd for distributed services

**Observability & Monitoring:**
- **Primary**: Langfuse for LLM tracing, metrics, and observability
- **Logging**: OpenTelemetry with structured logging
- **Analytics**: ClickHouse for query analytics and performance metrics

### Key Features

**Core RAG Functionality:**
- **Document Ingestion**: Automated PDF processing and chunking (1000 tokens, 200 overlap)
- **Semantic Search**: Vector similarity search with FAISS/Milvus
- **Context-Aware Generation**: RAG pipeline with source attribution
- **Streaming Responses**: Real-time response generation with SSE
- **Conversation Context**: Session-based conversation memory

**Advanced Features:**
- **Multi-Modal Processing**: PDF text extraction with layout preservation
- **Intelligent Chunking**: Content-aware document segmentation
- **Source Citation**: Automatic reference linking to source documents
- **Query Cancellation**: User-controllable request termination
- **Content Caching**: Hash-based vector store caching for efficiency

**Enterprise Features:**
- **Distributed Processing**: Airflow DAGs for scalable document processing
- **Feature Store**: Feast integration for ML feature management
- **Monitoring**: End-to-end pipeline observability with Langfuse
- **API Gateway**: NGINX load balancing and routing
- **Health Checks**: Comprehensive service health monitoring

## Data Sources
- **Official Documentation**: USCIS, Department of State, SEVP resources
- **University Policies**: Institution-specific international student guidelines
- **Legal Documents**: I-20, DS-2019, visa application examples
- **Regulatory Updates**: Current OPT/CPT guidelines and policy changes
- **Case Studies**: Anonymized examples of common visa scenarios

## Deployment Strategy

### Current Implementation
1. **Local Development**: Docker Compose orchestration with full stack
2. **Cloud Production**: GKE deployment with Helm charts
3. **CI/CD Pipeline**: Automated testing and deployment workflows
4. **Monitoring**: Langfuse integration for production observability

### Infrastructure Components
- **Compute**: RunPod GPU instances for LLM inference
- **Storage**: GKE persistent volumes with MinIO object storage
- **Networking**: NGINX ingress controller with SSL termination
- **Security**: Kubernetes secrets management and RBAC

## Success Metrics

**User Experience:**
- Query response time < 3 seconds (95th percentile)
- User satisfaction rating > 4.5/5.0
- Query resolution rate > 90%
- Session completion rate > 80%

**Technical Performance:**
- System uptime > 99.5%
- Vector retrieval latency < 500ms
- LLM inference time < 2 seconds
- Document processing throughput > 100 pages/hour

**Business Impact:**
- Monthly active users growth
- Average session duration and query depth
- Document corpus coverage and freshness
- Cost per query optimization

## Technical Implementation Phases

### Phase 1: Infrastructure Consolidation ✅
- ✅ Migrated monitoring from Prometheus/Grafana/Jaeger to Langfuse
- ✅ Integrated RunPod for scalable LLM inference
- ✅ Implemented Airflow data pipeline with Milvus/Feast
- ✅ Established Docker Compose development environment

### Phase 2: Production Readiness (Current)
- **Kubernetes Migration**: Clean rebuild of K8s deployment
- **Pipeline Optimization**: Airflow DAG performance tuning
- **Monitoring Enhancement**: Complete Langfuse integration
- **Security Hardening**: Secrets management and access controls

### Phase 3: Advanced Features
- **Multi-User Support**: Session management and user profiles
- **Advanced RAG**: Hybrid search with keyword + semantic
- **Document Versioning**: Automated corpus updates
- **Performance Optimization**: Caching and query optimization

## Next Steps - Immediate Actions Required

### 1. Kubernetes Infrastructure Rebuild
**Priority**: High
- **Action**: Complete redesign of Kubernetes deployment
- **Components**: Clean Helm charts, proper resource allocation, service mesh
- **Timeline**: 2-3 weeks

### 2. Airflow Pipeline Optimization
**Priority**: High
- **Action**: Optimize data ingestion and embedding pipeline
- **Focus**: Parallel processing, error handling, monitoring
- **Timeline**: 1-2 weeks

### 3. Langfuse Integration Completion
**Priority**: Medium
- **Action**: Full migration from legacy monitoring stack
- **Components**: Tracing, metrics collection, dashboard configuration
- **Timeline**: 1 week

### 4. RunPod Integration Testing
**Priority**: Medium
- **Action**: Comprehensive testing of RunPod inference pipeline
- **Focus**: Performance, reliability, cost optimization
- **Timeline**: 1 week

### 5. Feature Store Enhancement
**Priority**: Medium
- **Action**: Optimize Feast configuration for production workloads
- **Focus**: Feature serving performance, data freshness
- **Timeline**: 1-2 weeks

### 6. Documentation and Testing
**Priority**: Medium
- **Action**: Comprehensive documentation and test coverage
- **Components**: API docs, deployment guides, integration tests
- **Timeline**: 1-2 weeks

## Risk Management

**Technical Risks:**
- **RunPod Dependency**: Implement OpenAI fallback and local model support
- **Vector Store Performance**: Monitor Milvus scaling and optimize queries
- **Data Pipeline Failures**: Robust error handling and retry mechanisms

**Operational Risks:**
- **Cost Management**: Monitor RunPod GPU usage and optimize batch processing
- **Security**: Implement proper authentication and data protection
- **Compliance**: Ensure FERPA compliance for student data handling

**Mitigation Strategies:**
- Comprehensive monitoring with Langfuse
- Automated testing and deployment pipelines
- Clear documentation and incident response procedures
- Regular security audits and dependency updates

## Future Expansion
- **Multi-Language Support**: Spanish, Mandarin, Hindi language models
- **University Integration**: SSO and student information system APIs
- **Mobile Application**: React Native or Flutter mobile client
- **Advanced Analytics**: Student success correlation analysis
- **Compliance Automation**: Automated form filling and deadline tracking

## Constraints and Limitations
- **Legal Disclaimer**: Information only, not legal advice
- **Data Freshness**: Dependent on manual document updates
- **Model Limitations**: Subject to LLM hallucination and knowledge cutoffs
- **Cost Optimization**: Balance between performance and inference costs
- **Regulatory Compliance**: Must adhere to educational data protection requirements

This PRD reflects the current sophisticated architecture with RunPod GPU hosting, Langfuse observability, and comprehensive MLOps practices, positioning OPT-RAG as an enterprise-ready solution for international student support.