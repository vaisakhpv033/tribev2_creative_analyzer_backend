from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import api
from database import engine
import models

# Initialize Database tables
# Schema creation is handled by Alembic migrations
# models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Creative Quality Scorer API",
    description="Backend API for analyzing ad video creatives using TRIBEv2.",
    version="1.0.0",
)

import os

# CORS configuration
# You can set multiple origins by separating them with commas
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost,http://localhost:3000")
origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Creative Quality Scorer API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
