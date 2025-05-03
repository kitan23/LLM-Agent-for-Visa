# NGINX API Gateway for OPT-RAG

This document explains how the NGINX API Gateway is configured for the OPT-RAG International Student Visa Assistant project.

## What is an API Gateway?

An API Gateway serves as a single entry point for all client requests, routing them to appropriate backend services. It provides several benefits:

- **Unified Entry Point**: All requests go through a single point, simplifying client interactions
- **Load Balancing**: Distributes traffic across multiple backend instances
- **Security**: Adds a layer of protection for backend services
- **Rate Limiting**: Prevents abuse by limiting request rates
- **Monitoring**: Centralizes request logging
- **SSL/TLS Termination**: Handles encryption/decryption, reducing load on backend services

## How OPT-RAG Uses NGINX

The OPT-RAG system uses NGINX as both an API Gateway and a Load Balancer:

### 1. API Gateway Functionality

NGINX routes requests to appropriate services based on URL paths:

- `/api/*` → FastAPI backend (OPT-RAG API)
- `/` → Streamlit UI
- `/metrics` → Prometheus metrics (access restricted)
- `/health` → Health check endpoint
- `/static/*` → Static files

### 2. Load Balancing

NGINX distributes traffic to multiple instances of the same service using the `upstream` directive:

```nginx
upstream opt_rag_backend {
    server opt-rag-api:8000;
    # Add more servers for horizontal scaling
    # server opt-rag-api-2:8000;
    # server opt-rag-api-3:8000;
}
```

For horizontal scaling, simply uncomment the additional server lines and launch more instances.

## Key Configuration Elements

### 1. Rate Limiting

To prevent abuse, NGINX implements rate limiting:

```nginx
# Define the rate limit zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# Apply rate limiting to API endpoints
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    # ...
}
```

This configuration:
- Limits each client IP to 10 requests per second on average
- Allows bursts of up to 20 requests
- Stores client data in a 10MB memory zone

### 2. Request Rewriting

NGINX rewrites API requests to remove the `/api` prefix before forwarding to backend:

```nginx
location /api/ {
    rewrite ^/api/(.*) /$1 break;
    proxy_pass http://opt_rag_backend;
    # ...
}
```

For example, a client request to `/api/query` becomes `/query` when it reaches the backend server.

### 3. Proxy Headers

NGINX adds headers to forwarded requests:

```nginx
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

These headers help backend services understand:
- The original client IP address
- The protocol used (HTTP vs HTTPS)
- The original host requested

### 4. WebSocket Support

For Streamlit's live updates, WebSocket support is enabled:

```nginx
location / {
    # ...
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### 5. Security Headers

NGINX adds security headers to all responses:

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
```

## Scaling the System

To scale the OPT-RAG system horizontally:

1. Add more backend instances to the `upstream` block in nginx.conf
2. Launch additional API containers with different names (opt-rag-api-2, opt-rag-api-3, etc.)
3. Each instance should share the same vector store (via mounted volume) or use a distributed vector database

## SSL/TLS Configuration (Production)

For production deployment, enable HTTPS:

1. Uncomment the HTTPS redirect server block
2. Add your SSL certificates
3. Update the `server_name` directive with your domain

## Monitoring and Logging

NGINX logs are stored in:
- `/var/log/nginx/access.log` - Request logs
- `/var/log/nginx/error.log` - Error logs

These logs are mapped to `./nginx/logs/` on the host for easy access.

## Troubleshooting

### Common Issues

1. **502 Bad Gateway**: Backend server is unreachable
   - Check if the API service is running
   - Verify network connectivity between containers

2. **504 Gateway Timeout**: Backend server took too long to respond
   - Increase timeout settings
   - Investigate backend performance issues

3. **Rate limiting errors (429)**: Too many requests
   - Adjust rate limiting settings if legitimate traffic is blocked

### Useful Commands

```bash
# Check NGINX configuration syntax
docker exec opt-rag_nginx_1 nginx -t

# Reload NGINX configuration without restarting
docker exec opt-rag_nginx_1 nginx -s reload

# View NGINX logs
docker exec opt-rag_nginx_1 tail -f /var/log/nginx/error.log
``` 