from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class TaskType(str, Enum):
    PREDICT = "predict"
    PREDICT_FULL = "predict_full"

class WSMessage(BaseModel):
    """Base WebSocket message envelope."""
    type: str
    task_id: str

class WSProgress(WSMessage):
    """Partial result streamed during inference."""
    type: str = "progress"
    current: int
    total: int
    data: dict  # partial result payload

class WSComplete(WSMessage):
    """Final result when inference finishes."""
    type: str = "complete"
    data: dict

class WSError(WSMessage):
    """Error during inference."""
    type: str = "error"
    detail: str

class WSCancelled(WSMessage):
    """Confirmation that a task was cancelled."""
    type: str = "cancelled"

# REST request / response

class InferenceRequest(BaseModel):
    """Request to start an inference task."""
    task_type: TaskType
    # CSV data as string (uploaded via multipart in real usage,
    # but string is easier for programmatic testing)
    input_csv: Optional[str] = None
    query_csv: Optional[str] = None
    n_drugs: int = Field(default=3, ge=1, le=3)
    sample_ids: Optional[list[str]] = None  # filter to specific samples

class TaskInfo(BaseModel):
    """Status of a submitted task."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress_current: int = 0
    progress_total: int = 0
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"

class ControlConfig(BaseModel):
    """Control sample configuration."""
    method: str = "median"  # "user_selected" | "median"
    control_sample_ids: list[str] = Field(default_factory=list)

class SetControlsRequest(BaseModel):
    """Request to set control samples."""
    control_sample_ids: list[str]

class DeltaRequest(BaseModel):
    """Request to compute deltas from a completed prediction task."""
    task_id: str

class DeltaResponse(BaseModel):
    """Delta computation result."""
    deltas: list[dict]
    method: str
    control_sample_ids: list[str]

class UploadResponse(BaseModel):
    """Response after uploading a CSV."""
    row_count: int
    columns: list[str]
    column_mapping: dict[str, str]  # {role: column_name}
    degree: int = 1
    has_viability: bool = False
    dose_scale: str = "linear"      # "linear" or "log"
    dose_transform_applied: str = "none"  # "log10" or "none"
    warnings: list[str] = Field(default_factory=list)
    sample_ids: list[str] = Field(default_factory=list)
    csv_text: Optional[str] = None

class ColumnMappingOverride(BaseModel):
    """Request to override column mapping."""
    sample_id: Optional[str] = None
    drug1: Optional[str] = None
    dose1: Optional[str] = None
    drug2: Optional[str] = None
    dose2: Optional[str] = None
    drug3: Optional[str] = None
    dose3: Optional[str] = None
    viability: Optional[str] = None
    group: Optional[str] = None

class DrugInfo(BaseModel):
    """Information about a single drug."""
    name: str
    known: bool
    canonical_name: Optional[str] = None

class DrugMappingResponse(BaseModel):
    """Response with drug library mapping for all drugs in the data."""
    drugs: list[DrugInfo]

class QueryGenerateRequest(BaseModel):
    """Request to generate a query dataset."""
    degree: Optional[int] = None  # None = auto-infer from input
    n_points: int = Field(default=10, ge=1, le=1000)
    dose_min: Optional[float] = None  # log10 scale, default -6
    dose_max: Optional[float] = None  # log10 scale, default 4

class QueryGenerateResponse(BaseModel):
    """Response with generated query info and preview."""
    row_count: int
    degree: int
    preview: list[dict]  # First ~20 rows as dicts

class CohortInfo(BaseModel):
    """Metadata for a pre-built cohort."""
    name: str
    description: str
    n_samples: int
    n_drugs: int

class CohortListResponse(BaseModel):
    """Response listing available cohorts."""
    cohorts: list[CohortInfo]
