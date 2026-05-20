from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

class NeuralScoreBase(BaseModel):
    overall_score: Optional[float] = None
    visual_score: Optional[float] = None
    auditory_score: Optional[float] = None
    emotional_score: Optional[float] = None
    attention_score: Optional[float] = None
    language_score: Optional[float] = None

class VideoResponse(BaseModel):
    id: UUID
    filename: str
    original_name: str
    upload_time: datetime
    status: str
    job_id: Optional[str] = None
    scores: Optional[NeuralScoreBase] = None
    
    class Config:
        from_attributes = True
