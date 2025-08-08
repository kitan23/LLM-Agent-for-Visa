# OPT-RAG Implementation Tasks

## Overview
This document outlines the tasks needed to complete all course requirements while keeping the implementation simple and avoiding unnecessary complexity.

## Current Status Checklist

### ✅ Completed Components
- [x] **LLM Inference Layer (10%)**: RunPod + local models
- [x] **Data Ingestion Pipeline (10%)**: Airflow with load, chunk, embed steps
- [x] **Observability Layer (10%)**: Langfuse integration
- [x] **Storage Technologies**: PostgreSQL, Milvus, MinIO, Redis, Feast

### ⚠️ Missing/Incomplete Components
- [ ] **Guardrails Layer (10%)**
- [ ] **Fine-tuning Pipeline (10%)**
- [ ] **Routing Layer (10%)** - Needs enhancement
- [ ] **Gateway Layer (10%)**
- [ ] **CI/CD (10%)**
- [ ] **Comprehensive Monitoring (10%)** - Needs logs & metrics

---

## Implementation Tasks

### 1. Guardrails Layer Implementation
**Goal**: Add input/output validation without external dependencies

#### Task 1.1: Create Guardrails Middleware
- **Location**: `rag-pipeline/src/guardrails/validator.py`
- **Implementation**:
  ```python
  # Simple validation class with:
  - Input length validation (max tokens)
  - Prompt injection detection (regex patterns)
  - Output content filtering (PII, inappropriate content)
  - Rate limiting per session
  ```
- **Integration Point**: Add to FastAPI middleware in `main.py`
- **No new dependencies**: Use built-in Python libraries only

#### Task 1.2: Add Validation Rules Configuration
- **Location**: `rag-pipeline/config/guardrails.yaml`
- **Content**:
  - Max input length: 2000 tokens
  - Blocked patterns list
  - Rate limits: 10 queries/minute
  - PII regex patterns

---

### 2. Fine-tuning Pipeline (Mock Implementation)
**Goal**: Demonstrate fine-tuning capability without actual model training

#### Task 2.1: Create Fine-tuning Airflow DAG
- **Location**: `infra/dags/fine_tuning_pipeline.py`
- **Steps**:
  1. **prepare_dataset**: Load sample Q&A pairs from CSV
  2. **mock_fine_tuning**: Simulate training (sleep + progress logs)
  3. **offline_evaluation**: Calculate mock metrics (accuracy, F1)
  4. **save_model_metadata**: Store "results" in PostgreSQL

#### Task 2.2: Add Evaluation Metrics Storage
- **Location**: `infra/plugins/jobs/evaluation.py`
- **Implementation**:
  - Mock BLEU/ROUGE scores calculation
  - Store results in PostgreSQL table
  - Generate simple evaluation report

---

### 3. Routing Layer Enhancement
**Goal**: Improve existing routing with clear fallback logic

#### Task 3.1: Enhance Routing Logic
- **Location**: `rag-pipeline/src/llm/router.py`
- **Implementation**:
  ```python
  class LLMRouter:
      - Primary: RunPod API (if available)
      - Secondary: OpenAI API (if RunPod fails)
      - Tertiary: Local model (if both fail)
      - Health check endpoint for each service
      - Simple round-robin for load distribution
  ```

#### Task 3.2: Add Routing Configuration
- **Location**: `rag-pipeline/config/routing.yaml`
- **Content**:
  - Service priorities
  - Timeout settings
  - Retry policies
  - Health check intervals

---

### 4. Gateway Layer (Simple NGINX)
**Goal**: Add API gateway without complex service mesh

#### Task 4.1: Add NGINX to Docker Compose
- **Location**: Update `docker-compose.yml`
- **Configuration**:
  ```yaml
  nginx:
    image: nginx:alpine
    volumes:
      - ./infra/nginx/nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - rag-api
  ```

#### Task 4.2: Create NGINX Configuration
- **Location**: `infra/nginx/nginx.conf`
- **Features**:
  - Rate limiting (10 req/s)
  - Request size limit (1MB)
  - Basic authentication (optional)
  - Reverse proxy to FastAPI
  - Access logs

---

### 5. CI/CD Implementation
**Goal**: Simple GitHub Actions workflow

#### Task 5.1: Create GitHub Actions Workflow
- **Location**: `.github/workflows/ci.yml`
- **Steps**:
  1. Run linting (ruff/black)
  2. Run unit tests (pytest)
  3. Build Docker images
  4. Run integration tests
  5. Deploy notification (webhook)

#### Task 5.2: Add Basic Tests
- **Location**: `tests/`
- **Coverage**:
  - Test guardrails validation
  - Test routing logic
  - Test API endpoints
  - Test Airflow DAGs validation

---

### 6. Comprehensive Monitoring Enhancement
**Goal**: Complete logs, metrics, traces setup

#### Task 6.1: Structured Logging
- **Location**: Update `rag-pipeline/src/utils/logging.py`
- **Implementation**:
  - JSON formatted logs
  - Log aggregation to file
  - Log levels configuration
  - Correlation IDs for request tracking

#### Task 6.2: Add Metrics Collection
- **Location**: `rag-pipeline/src/utils/metrics.py`
- **Metrics**:
  - Request count
  - Response time
  - Model inference time
  - Cache hit rate
  - Error rate
- **Export**: Prometheus format endpoint `/metrics`

#### Task 6.3: Enhance Langfuse Integration
- **Current**: Already integrated
- **Enhancement**:
  - Add custom events
  - Track user sessions
  - Monitor token usage
  - Track cache performance

---

## Implementation Order & Priority

### Phase 1 (Core Requirements) - Week 1
1. **Guardrails Layer** (2 days)
2. **Gateway Layer** (1 day)
3. **Routing Enhancement** (1 day)

### Phase 2 (Pipeline & Monitoring) - Week 2
4. **Fine-tuning Pipeline** (2 days)
5. **Monitoring Enhancement** (2 days)

### Phase 3 (Automation) - Week 3
6. **CI/CD Setup** (2 days)
7. **Testing & Documentation** (2 days)

---

## Key Principles
1. **No New Heavy Dependencies**: Use existing tech stack
2. **Mock When Appropriate**: Fine-tuning can be simulated
3. **Configuration Over Code**: Use YAML configs for flexibility
4. **Incremental Changes**: Don't break existing functionality
5. **Simple Solutions**: Avoid enterprise patterns for personal project

---

## Files to Create/Modify

### New Files
- `rag-pipeline/src/guardrails/validator.py`
- `rag-pipeline/src/llm/router.py`
- `rag-pipeline/config/guardrails.yaml`
- `rag-pipeline/config/routing.yaml`
- `infra/dags/fine_tuning_pipeline.py`
- `infra/plugins/jobs/evaluation.py`
- `infra/nginx/nginx.conf`
- `.github/workflows/ci.yml`
- `tests/test_*.py`

### Files to Modify
- `docker-compose.yml` (add nginx service)
- `rag-pipeline/src/main.py` (add guardrails middleware)
- `rag-pipeline/src/utils/logging.py` (structured logging)
- `rag-pipeline/src/utils/metrics.py` (create new)

---

## Success Criteria
- [ ] All 6 missing components implemented
- [ ] Docker Compose runs without errors
- [ ] All course requirements satisfied (100% score)
- [ ] No increase in deployment complexity
- [ ] Documentation updated
- [ ] Basic tests passing