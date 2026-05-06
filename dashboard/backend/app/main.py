from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from .inference import InferenceWorker
from .session_store import session_store
from .schemas import (
    ControlConfig,
    DeltaRequest,
    DeltaResponse,
    HealthResponse,
    InferenceRequest,
    SetControlsRequest,
    TaskInfo,
    TaskStatus,
    TaskType,
)

logger = logging.getLogger(__name__)

_toxicity_db: dict | None = None

def _load_toxicity_db() -> dict:
    global _toxicity_db
    if _toxicity_db is not None:
        return _toxicity_db
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "toxicity", "toxicity_db.json")
    if os.path.exists(db_path):
        with open(db_path) as f:
            _toxicity_db = json.load(f)
    else:
        _toxicity_db = {"drugs": {}, "combinations": {}}
    return _toxicity_db

MODEL_DIR = os.environ.get(
    "SCREENSHOT_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "model_batchie"),
)
MODEL_CHECKPOINT = os.environ.get(
    "SCREENSHOT_MODEL_CHECKPOINT",
    os.path.join(MODEL_DIR, "final_model"),
)
DEVICE = os.environ.get("SCREENSHOT_DEVICE", "auto")

worker: Optional[InferenceWorker] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None  # Captured at startup for thread-safe WS sends
ws_connections: dict[str, list[WebSocket]] = {}

def _progress_callback(task_id: str, current: int, total: int, data: dict) -> None:
    """Called from inference thread — schedules WS sends on the main event loop."""
    conns = ws_connections.get(task_id, [])
    if not conns or _event_loop is None:
        return
    msg = json.dumps({
        "type": "progress",
        "task_id": task_id,
        "current": current,
        "total": total,
        "data": data,
    })
    for ws in conns:
        _event_loop.call_soon_threadsafe(asyncio.ensure_future, ws.send_text(msg))

def _completion_callback(task_id: str, status: str, result: dict | None, error: str | None) -> None:
    """Called from inference thread when a task completes — sends final WS message."""
    conns = ws_connections.get(task_id, [])
    if not conns or _event_loop is None:
        return
    final_msg: dict = {
        "type": status,
        "task_id": task_id,
    }
    if status == "completed" and result:
        final_msg["data"] = result
    if status == "failed" and error:
        final_msg["detail"] = error
    msg = json.dumps(final_msg)
    try:
        for ws in conns:
            _event_loop.call_soon_threadsafe(asyncio.ensure_future, ws.send_text(msg))
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, _event_loop
    _event_loop = asyncio.get_running_loop()
    worker = InferenceWorker(
        model_checkpoint=MODEL_CHECKPOINT,
        model_dir=MODEL_DIR,
        device=DEVICE,
    )
    worker.on_progress(_progress_callback)
    worker.on_completion(_completion_callback)
    await worker.start()
    yield
    await worker.stop()

app = FastAPI(
    title="ScreenShot Dashboard API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routers import upload, drugs, query_gen, cohorts

app.include_router(upload.router)
app.include_router(drugs.router)
app.include_router(query_gen.router)
app.include_router(cohorts.router)

# REST endpoints

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=worker.model_loaded if worker else False,
        device=worker.actual_device if worker else "not_started",
    )

@app.post("/tasks", response_model=TaskInfo)
async def create_task(req: InferenceRequest):
    """Submit a new inference task."""
    params = req.model_dump(exclude={"task_type"})
    task_id = worker.submit(req.task_type, params)
    task = worker.get_task(task_id)
    return task.to_info()

async def _read_upload_as_csv(upload: UploadFile) -> str:
    raw = await upload.read()
    name = (upload.filename or "").lower()
    if name.endswith(".feather"):
        import io, pandas as pd
        return pd.read_feather(io.BytesIO(raw)).to_csv(index=False)
    if name.endswith((".xlsx", ".xls")):
        import io, pandas as pd
        return pd.read_excel(io.BytesIO(raw)).to_csv(index=False)
    return raw.decode("utf-8")

@app.post("/tasks/upload", response_model=TaskInfo)
async def create_task_upload(
    task_type: TaskType = Form(...),
    n_drugs: int = Form(3),
    input_file: UploadFile = File(...),
    query_file: Optional[UploadFile] = File(None),
):
    """Submit inference task with file upload (CSV or Feather)."""
    input_csv = await _read_upload_as_csv(input_file)
    query_csv = None
    if query_file is not None:
        query_csv = await _read_upload_as_csv(query_file)

    params = {
        "input_csv": input_csv,
        "query_csv": query_csv,
        "n_drugs": n_drugs,
    }
    task_id = worker.submit(task_type, params)
    task = worker.get_task(task_id)
    return task.to_info()

@app.post("/combination/predict")
async def combination_predict(request: Request):
    """Predict viability surface for a drug pair on a given sample."""
    import numpy as np
    import pandas as pd
    from io import StringIO

    body = await request.json()
    sample_id = body["sample_id"]
    drug1 = body["drug1"]
    drug2 = body["drug2"]
    n_points = body.get("n_points", 10)

    if not worker.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = session_store.input_meta
    if meta is None:
        raise HTTPException(status_code=400, detail="No input data uploaded")
    doses = np.linspace(-5, 3, n_points).tolist()
    query_rows = []
    for d1 in doses:
        for d2 in doses:
            query_rows.append({
                "sample_id": sample_id,
                "drug1": drug1, "dose1": 10.0 ** d1,
                "drug2": drug2, "dose2": 10.0 ** d2,
                "float_value": 1.0,
            })
    query_df = pd.DataFrame(query_rows)
    input_df = meta.standardized_df()
    input_df = session_store.dose_filter_df(input_df)
    input_df = session_store.subsample_df(input_df)

    sample_rows = len(input_df[input_df["sample_id"] == sample_id])
    print(f"Combination: sample={sample_id}, context={sample_rows}, subsample_n={session_store.subsample_n}, query={n_points}x{n_points}", flush=True)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, worker.predict_combination, input_df, query_df, sample_id)

    return {
        "doses": doses,  # log10 scale
        "viability": result,
        "drug1": drug1,
        "drug2": drug2,
        "sample_id": sample_id,
    }

@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    task = worker.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_info()

@app.post("/tasks/{task_id}/cancel", response_model=TaskInfo)
async def cancel_task(task_id: str):
    """Cancel a running or pending task."""
    found = worker.cancel(task_id)
    if not found:
        raise HTTPException(status_code=404, detail="Task not found")
    task = worker.get_task(task_id)
    return task.to_info()

@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """Get the full result of a completed task."""
    task = worker.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Task status is {task.status}")
    return task.result

@app.get("/controls", response_model=ControlConfig)
async def get_controls():
    """Get current control configuration."""
    return worker.get_controls()

@app.post("/controls", response_model=ControlConfig)
async def set_controls(req: SetControlsRequest):
    """Set control sample IDs."""
    return worker.set_controls(req.control_sample_ids)

@app.post("/compute/deltas")
async def compute_deltas(req: DeltaRequest):
    """Compute deltas from a completed prediction task."""
    result = worker.compute_deltas(req.task_id)
    if result is None:
        raise HTTPException(status_code=409, detail="Task not found")
    if result == "not_completed":
        raise HTTPException(status_code=409, detail="Task not completed yet")
    return result

@app.post("/inference/start", response_model=TaskInfo)
async def start_inference():
    from .session_store import session_store

    input_df = session_store.input_df
    query_df = session_store.query_df

    if input_df is None:
        raise HTTPException(status_code=400, detail="No input data uploaded yet.")
    if query_df is None:
        raise HTTPException(status_code=400, detail="No query data available. Upload or generate a query first.")

    degree = session_store.input_meta.degree if session_store.input_meta else 1
    input_df = session_store.dose_filter_df(input_df)
    input_df = session_store.subsample_df(input_df)

    input_csv = input_df.to_csv(index=False)
    query_csv = query_df.to_csv(index=False)

    params = {
        "input_csv": input_csv,
        "query_csv": query_csv,
        "n_drugs": min(degree, 3),
    }
    task_id = worker.submit(TaskType.PREDICT, params)
    task = worker.get_task(task_id)
    return task.to_info()

@app.post("/inference/start-mae", response_model=TaskInfo)
async def start_mae_inference():
    from .session_store import session_store

    input_df = session_store.input_df
    if input_df is None:
        raise HTTPException(status_code=400, detail="No input data uploaded yet.")

    if "float_value" not in input_df.columns:
        raise HTTPException(status_code=400, detail="Input data has no viability column — cannot compute MAE.")

    degree = session_store.input_meta.degree if session_store.input_meta else 1
    full_input_csv = input_df.to_csv(index=False)

    params = {
        "input_csv": full_input_csv,
        "query_csv": full_input_csv,  # Query = input → predictions get float_value
        "n_drugs": min(degree, 3),
    }
    task_id = worker.submit(TaskType.PREDICT, params)
    task = worker.get_task(task_id)
    return task.to_info()

@app.post("/session/reset")
async def reset_session():
    session_store.clear()
    return {"status": "ok"}

@app.get("/device")
async def get_device():
    import torch
    available = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available.append("mps")
    if torch.cuda.is_available():
        available.append("cuda")
    return {
        "current": worker.device_name,
        "available": available,
    }

@app.post("/device")
async def set_device(request: Request):
    body = await request.json()
    device = body.get("device", "cpu")

    await worker.stop()
    worker.device = device
    await worker.start()

    return {"current": worker.device_name}

@app.get("/session/subsample")
async def get_subsample():
    return {
        "n_per_sample": session_store.subsample_n,
        "selected_indices": session_store.get_subsample_indices(),
    }

@app.post("/session/subsample")
async def set_subsample(request: Request):
    body = await request.json()
    n = body.get("n_per_sample")
    if n is not None:
        n = int(n)
        if n < 1:
            raise HTTPException(status_code=400, detail="n_per_sample must be >= 1")
    session_store.subsample_n = n
    return {
        "n_per_sample": session_store.subsample_n,
        "selected_indices": session_store.get_subsample_indices(),
    }

@app.get("/session/dose-filter")
async def get_dose_filter():
    return {
        "selected_doses": session_store.dose_filter,
        "available_doses": session_store.get_available_doses(),
        "selected_indices": session_store.get_dose_filter_indices(),
    }

@app.post("/session/dose-filter")
async def set_dose_filter(request: Request):
    body = await request.json()
    doses = body.get("selected_doses")
    if doses is not None:
        doses = [float(d) for d in doses]
        if len(doses) == 0:
            doses = None
    session_store.dose_filter = doses
    return {
        "selected_doses": session_store.dose_filter,
        "available_doses": session_store.get_available_doses(),
        "selected_indices": session_store.get_dose_filter_indices(),
    }

@app.get("/session/restore")
async def restore_session():
    if session_store.input_meta is None:
        return {"has_input": False}

    meta = session_store.input_meta
    mapping_dict = meta.column_mapping.to_dict() if meta.column_mapping else {}
    input_csv = meta.raw_df.to_csv(index=False)

    has_query = session_store.query_df is not None

    return {
        "has_input": True,
        "upload_response": {
            "row_count": meta.row_count,
            "columns": meta.columns,
            "column_mapping": mapping_dict,
            "degree": meta.degree,
            "has_viability": meta.has_viability,
            "dose_scale": meta.dose_scale,
            "dose_transform_applied": meta.dose_transform_applied,
            "warnings": meta.warnings,
        },
        "input_csv": input_csv,
        "has_query": has_query,
        "subsample_n": session_store.subsample_n,
        "dose_filter": session_store.dose_filter,
    }

@app.get("/session/groups")
async def get_session_groups():
    from .session_store import session_store

    if session_store.input_meta is None:
        return {"groups": {}, "group_column": None}

    mapping = session_store.input_meta.column_mapping
    if mapping is None or mapping.group is None:
        return {"groups": {}, "group_column": None}

    df = session_store.input_meta.raw_df
    group_col = mapping.group
    sample_col = mapping.sample_id

    if group_col not in df.columns or sample_col not in df.columns:
        return {"groups": {}, "group_column": group_col}
    groups = {}
    for _, row in df.drop_duplicates(subset=[sample_col]).iterrows():
        groups[str(row[sample_col])] = str(row[group_col])

    return {"groups": groups, "group_column": group_col}

@app.get("/data/status")
async def data_status():
    """Check if input and query data are available for inference."""
    has_input = session_store.input_df is not None
    has_query = session_store.query_df is not None
    n_input_rows = len(session_store.input_df) if has_input else 0
    n_query_rows = len(session_store.query_df) if has_query else 0
    return {
        "has_input": has_input,
        "has_query": has_query,
        "n_input_rows": n_input_rows,
        "n_query_rows": n_query_rows,
    }

@app.get("/toxicity/combo")
async def get_combo_toxicity(drugs: str):
    """Look up combination toxicity. drugs=DrugA,DrugB"""
    db = _load_toxicity_db()
    drug_names = [d.strip().lower() for d in drugs.split(",") if d.strip()]
    individual = []
    for name in drug_names:
        entry = db["drugs"].get(name)
        if entry:
            individual.append(entry)
        else:
            individual.append({"name": name, "toxicity_grade": None, "message": "No data available"})
    combo_key = "+".join(sorted(drug_names))
    combo_info = db.get("combinations", {}).get(combo_key)

    return {
        "drugs": individual,
        "combination": combo_info,
    }

@app.get("/toxicity/{drug_name}")
async def get_toxicity(drug_name: str):
    """Look up toxicity info for a drug."""
    db = _load_toxicity_db()
    key = drug_name.strip().lower()
    entry = db["drugs"].get(key)
    if entry:
        return entry
    return {"name": drug_name, "toxicity_grade": None, "message": "No toxicity data available for this drug."}

@app.get("/session/data")
async def get_session_data():
    input_df = session_store.input_df
    query_df = session_store.query_df

    if input_df is None:
        raise HTTPException(status_code=400, detail="No input data uploaded yet.")

    input_csv = input_df.to_csv(index=False)
    query_csv = query_df.to_csv(index=False) if query_df is not None else None

    return {
        "input_csv": input_csv,
        "query_csv": query_csv,
        "n_samples": input_df["sample_id"].nunique() if "sample_id" in input_df.columns else 0,
    }

@app.websocket("/ws/{task_id}")
async def websocket_task(websocket: WebSocket, task_id: str):
    """Stream inference progress for a specific task."""
    await websocket.accept()
    if task_id not in ws_connections:
        ws_connections[task_id] = []
    ws_connections[task_id].append(websocket)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                data = json.loads(msg)
                if data.get("action") == "cancel":
                    worker.cancel(task_id)
                    await websocket.send_text(json.dumps({
                        "type": "cancelled",
                        "task_id": task_id,
                    }))
            except asyncio.TimeoutError:
                task = worker.get_task(task_id)
                if task and task.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.CANCELLED,
                    TaskStatus.FAILED,
                ):
                    final_msg = {
                        "type": task.status.value,
                        "task_id": task_id,
                    }
                    if task.status == TaskStatus.COMPLETED and task.result:
                        final_msg["data"] = task.result
                    if task.status == TaskStatus.FAILED:
                        final_msg["detail"] = task.error
                    await websocket.send_text(json.dumps(final_msg))
                    break
    except WebSocketDisconnect:
        pass
    finally:
        ws_connections.get(task_id, []).remove(websocket) if websocket in ws_connections.get(task_id, []) else None

async def _monitor_task_completion(websocket: WebSocket, task_id: str):
    """Background coroutine that monitors a task and sends the final status."""
    try:
        while True:
            await asyncio.sleep(0.5)
            task = worker.get_task(task_id)
            if task is None:
                break
            if task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
                TaskStatus.FAILED,
            ):
                final_msg: dict = {
                    "type": task.status.value,
                    "task_id": task_id,
                }
                if task.status == TaskStatus.COMPLETED and task.result:
                    final_msg["data"] = task.result
                if task.status == TaskStatus.FAILED:
                    final_msg["detail"] = task.error
                try:
                    await websocket.send_text(json.dumps(final_msg))
                except Exception:
                    pass
                break
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

@app.websocket("/ws")
async def websocket_general(websocket: WebSocket):
    """General WebSocket for submitting tasks and receiving all updates."""
    await websocket.accept()

    monitor_tasks: list[asyncio.Task] = []

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            action = data.get("action")

            if action == "submit":
                task_type = TaskType(data["task_type"])
                params = {k: v for k, v in data.items() if k not in ("action", "task_type")}
                # when not explicitly provided or when use_session flag is set
                if data.get("use_session") or "input_csv" not in params or not params.get("input_csv"):
                    input_df = session_store.input_df
                    if input_df is not None:
                        input_df = session_store.dose_filter_df(input_df)
                        input_df = session_store.subsample_df(input_df)
                        params["input_csv"] = input_df.to_csv(index=False)
                if data.get("use_session") or "query_csv" not in params or not params.get("query_csv"):
                    query_df = session_store.query_df
                    if query_df is not None:
                        params["query_csv"] = query_df.to_csv(index=False)
                if session_store.input_meta and "n_drugs" not in params:
                    params["n_drugs"] = min(session_store.input_meta.degree, 3)

                task_id = worker.submit(task_type, params)
                if task_id not in ws_connections:
                    ws_connections[task_id] = []
                ws_connections[task_id].append(websocket)
                await websocket.send_text(json.dumps({
                    "type": "submitted",
                    "task_id": task_id,
                }))
                mt = asyncio.create_task(
                    _monitor_task_completion(websocket, task_id)
                )
                monitor_tasks.append(mt)

            elif action == "cancel":
                task_id = data.get("task_id")
                if task_id:
                    worker.cancel(task_id)
                    await websocket.send_text(json.dumps({
                        "type": "cancelled",
                        "task_id": task_id,
                    }))

    except WebSocketDisconnect:
        pass
    finally:
        for mt in monitor_tasks:
            mt.cancel()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        ws_max_size=16 * 1024 * 1024,  # 16 MB — predict_full sends large embedding payloads
    )
