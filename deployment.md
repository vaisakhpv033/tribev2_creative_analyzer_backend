# Deployment Guide: Creative Quality Scorer Backend

This document outlines the current deployment architecture of the Creative Quality Scorer Backend and provides step-by-step instructions for deploying it using Docker and Docker Compose. It also includes troubleshooting steps for common issues.

---

## 1. Architecture Overview

The backend is fully dockerized to ensure a consistent environment across development, staging, and production. The deployment architecture consists of four interconnected containers:

1. **API (`api`)**: The core FastAPI web server that handles incoming HTTP requests. It uses Uvicorn as the ASGI server.
2. **Worker (`worker`)**: A Celery background worker that processes heavy video analysis tasks asynchronously.
3. **Database (`db`)**: A PostgreSQL 15 database that stores application data.
4. **Redis (`redis`)**: A Redis 7 instance that serves two purposes:
   - **Message Broker**: Passes tasks from the FastAPI application to the Celery worker.
   - **Result Backend**: Stores the completion status and results of the Celery tasks.

### Automation
- **Migrations**: The API container uses a startup script (`start.sh`) that automatically runs database migrations (`alembic upgrade head`) before booting up the Uvicorn server. You do not need to manage schema migrations manually.

---

## 2. Prerequisites

To deploy this backend on any machine, you only need the following installed:
- **Docker**: Containerization engine.
- **Docker Compose**: Orchestration tool for multi-container applications.
- **Git**: To clone the repository.

---

## 3. Step-by-Step Deployment

Follow these steps to deploy the backend from scratch on a new server or local machine:

### Step 1: Clone the Repository
Clone the repository and navigate to the backend directory:
```bash
git clone https://github.com/vaisakhpv033/tribev2_creative_analyzer_backend.git
cd tribev2_creative_analyzer_backend/backend
```

### Step 2: Configure Environment Variables
A template for environment variables is provided in `.env.example`. 

Copy the example file to create your local `.env` file (if you are running the application outside of Docker in the future). 
```bash
cp .env.example .env
```
> **Note on Docker Compose**: The `docker-compose.yml` file is configured to *hardcode* the internal Docker network URLs for PostgreSQL and Redis to avoid conflicts with your local machine. However, if you have external API keys (like `TRIBEV2_API_BASE_URL`), you should set them in the `.env` file or export them in your host shell.

### Step 3: Build and Start the Containers
Run the following command to build the Docker images and start all containers in detached mode (`-d`):
```bash
docker-compose up --build -d
```
*Note: The initial build might take a few minutes because the backend uses heavy data science libraries like `xgboost`, `scipy`, and `nilearn` which require system-level C-compilers to install.*

### Step 4: Verify the Deployment
Once the containers are running, verify the API is up by checking the health endpoint. You can do this by visiting `http://localhost:8000/health` in your browser or running:
```bash
curl http://localhost:8000/health
```
**Expected Output:**
```json
{"status": "ok", "message": "Creative Quality Scorer API is running"}
```

---

## 4. Managing the Stack (Daily Workflow)

You **do not** need to rebuild the containers every time you stop them or change code. Here is a breakdown of the daily workflow commands:

### Normal Stopping and Starting
If you want to stop the backend at the end of the day and resume later, just pause and start the existing containers. This is instantaneous:
- **Stop**: `docker-compose stop`
- **Start**: `docker-compose start` (or `docker-compose up -d`)

### When You Modify Code (`main.py`, `worker.py`, etc.)
The `docker-compose.yml` mounts your local folder directly into the container using a volume (`volumes: - .:/app`). This means code changes are instantly visible to the container! You do **not** need to rebuild the image. Just restart the containers so they read the new code:
- **Restart**: `docker-compose restart api worker`

### When to actually use `--build`
You **only** need to run `docker-compose up --build -d` when you make changes to the infrastructure itself. Specifically, if you:
- Add a new Python package to your `requirements.txt`.
- Modify the `Dockerfile`.

### Essential Commands
- **View Logs**: To watch live logs from all containers:
  ```bash
  docker-compose logs -f
  ```
- **View Specific Container Logs**: To see logs for just the API or just the Worker:
  ```bash
  docker-compose logs -f api
  docker-compose logs -f worker
  ```
- **Tear Down the Stack**: Stop and remove all containers, networks, and images (Data volumes will **persist**):
  ```bash
  docker-compose down
  ```
- **Wipe Data Volumes**: If you want to completely wipe your Postgres database and Redis queues to start fresh:
  ```bash
  docker-compose down -v
  ```

---

## 5. Troubleshooting Common Issues

### Issue 1: `psycopg2.OperationalError: Connection refused`
**Symptom**: The API container crashes and the logs show it is trying to connect to `localhost:5432` instead of the `db` container.
**Cause**: The application is inadvertently loading a local `.env` file that overrides the Docker-provided `DATABASE_URL`.
**Resolution**: We have already configured `alembic/env.py` to use `load_dotenv(override=False)` and hardcoded the internal URLs in `docker-compose.yml`. Ensure you do not change `docker-compose.yml` to rely heavily on variable interpolation for internal hostnames like `db` or `redis` if you also keep a local `.env` file with `localhost` defined.

### Issue 2: Build Fails with "Out of Memory" (OOM) or Killed Process
**Symptom**: During `docker-compose build`, the process hangs on `pip install scipy` or `xgboost` and eventually dies.
**Cause**: Data science libraries require significant RAM to compile. Docker Desktop or your cloud server might not have enough memory allocated.
**Resolution**: 
- If using Docker Desktop (Windows/Mac), go to Settings -> Resources and increase Memory to at least 4GB or 8GB.
- If on a Linux server, ensure you have at least 4GB of RAM or configure a Swapfile.

### Issue 3: Port `8000`, `5432`, or `6379` is already allocated
**Symptom**: `docker-compose up` fails stating `bind: address already in use`.
**Cause**: You have another service (like a local Postgres or Redis installation) already running on your host machine using the required ports.
**Resolution**: 
1. Stop the local service running on that port.
2. Or, change the port mapping in `docker-compose.yml`. For example, change `"8000:8000"` to `"8080:8000"` to expose the API on port 8080 instead.

### Issue 4: Background tasks are stuck in "Pending"
**Symptom**: You submit a video for analysis, but it never completes. The task stays "Pending".
**Cause**: The Celery worker container might have crashed, or it lost connection to Redis.
**Resolution**: Check the worker logs using `docker-compose logs -f worker`. If it threw an error, restart the worker container using `docker-compose restart worker`.
