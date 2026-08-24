-- Creates the langfuse database required by the Langfuse services.
-- Runs once on first initialization of the postgres volume.
CREATE DATABASE langfuse;
GRANT ALL PRIVILEGES ON DATABASE langfuse TO airflow;
