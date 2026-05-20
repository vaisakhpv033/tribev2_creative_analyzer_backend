import uuid
import datetime
import enum
from sqlalchemy import Column, String, Float, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base

class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    INFERENCE = "INFERENCE"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, index=True)
    original_name = Column(String)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    job_id = Column(String, nullable=True) # RunPod Job ID
    npz_path = Column(String, nullable=True)
    
    scores = relationship("NeuralScore", back_populates="video", uselist=False)

class NeuralScore(Base):
    __tablename__ = "neural_scores"

    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), primary_key=True)
    overall_score = Column(Float, nullable=True)
    visual_score = Column(Float, nullable=True)
    auditory_score = Column(Float, nullable=True)
    emotional_score = Column(Float, nullable=True)
    attention_score = Column(Float, nullable=True)
    language_score = Column(Float, nullable=True)

    video = relationship("Video", back_populates="scores")

class BaselineCalibration(Base):
    __tablename__ = "baseline_calibration"

    dimension = Column(String, primary_key=True) # visual, auditory, emotional, attention, language
    min_value = Column(Float)
    max_value = Column(Float)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
