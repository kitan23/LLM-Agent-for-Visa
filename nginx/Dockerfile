# Use the official NGINX image as base
FROM nginx:1.25

# Copy our custom NGINX configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create a directory for static files
RUN mkdir -p /usr/share/nginx/html/static

# Add a sample static file for testing
RUN echo "OPT-RAG API Gateway" > /usr/share/nginx/html/static/info.txt

# Expose ports
EXPOSE 80
EXPOSE 443

# Set default command
CMD ["nginx", "-g", "daemon off;"] 