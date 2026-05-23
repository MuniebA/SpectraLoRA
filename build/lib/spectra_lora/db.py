import os
import json
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

"""
SpectraLoRA MLOps Database Tracker
----------------------------------
Automatically logs experiments to a local SQLite file, or connects to a 
PostgreSQL server if the SPECTRALORA_DB_URL environment variable is set.
"""

# 1. The Fallback Logic
DB_URL = os.getenv("SPECTRALORA_DB_URL", "sqlite:///spectralora_experiments.db")

# Create the Engine (Connects to DB)
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 2. Define the Schema (Tables & Columns)
class ExperimentRun(Base):
    __tablename__ = "spectralora_runs"
    
    # Run ID (Unique UUID string to prevent collisions)
    run_id = Column(String, primary_key=True, index=True)
    run_name = Column(String, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    # Metadata & Hardware
    device = Column(String) # CPU vs CUDA
    
    # Hyperparameters (Stored flexibly as JSON)
    model_config = Column(JSON)
    lora_config = Column(JSON)
    physics_config = Column(JSON)
    training_params = Column(JSON)
    
    # Final Outcomes
    weights_path = Column(String, nullable=True)
    status = Column(String, default="running") # "running", "completed", "failed"

class EpochMetric(Base):
    __tablename__ = "spectralora_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True) # Links back to ExperimentRun
    epoch = Column(Integer)
    
    # Core ML Metrics
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    
    # GeoAI Specific Metrics
    miou = Column(Float, nullable=True)
    pixel_accuracy = Column(Float, nullable=True)
    physics_violations = Column(Integer, nullable=True)

# 3. Auto-Generate the Schema (Creates tables if they don't exist)
Base.metadata.create_all(bind=engine)

# 4. Helper API Functions for train.py
def log_experiment_start(run_name: str, device: str, configs: dict) -> str:
    """Logs the start of a training run. Returns the unique run_id."""
    db = SessionLocal()
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    run = ExperimentRun(
        run_id=run_id,
        run_name=run_name,
        device=device,
        model_config=configs.get("model", {}),
        lora_config=configs.get("lora", {}),
        physics_config=configs.get("physics", {}),
        training_params=configs.get("training", {})
    )
    db.add(run)
    db.commit()
    db.close()
    return run_id

def log_epoch_metrics(run_id: str, epoch: int, metrics: dict):
    """Logs metrics for a single epoch."""
    db = SessionLocal()
    epoch_log = EpochMetric(
        run_id=run_id,
        epoch=epoch,
        train_loss=metrics.get("train_loss"),
        val_loss=metrics.get("val_loss"),
        learning_rate=metrics.get("learning_rate"),
        miou=metrics.get("miou"),
        pixel_accuracy=metrics.get("pixel_accuracy"),
        physics_violations=metrics.get("physics_violations")
    )
    db.add(epoch_log)
    db.commit()
    db.close()

def log_experiment_end(run_id: str, weights_path: str = None, status: str = "completed"):
    """Marks an experiment as finished and saves the weights path."""
    db = SessionLocal()
    run = db.query(ExperimentRun).filter(ExperimentRun.run_id == run_id).first()
    if run:
        run.end_time = datetime.utcnow()
        run.weights_path = weights_path
        run.status = status
        db.commit()
    db.close()