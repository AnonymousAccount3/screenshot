from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .controls import ControlState, compute_deltas as _compute_deltas_impl
from .metrics import compute_metrics_per_pair
from .schemas import TaskInfo, TaskStatus, TaskType

logger = logging.getLogger(__name__)

class CancellationToken:
    """Lightweight token checked between inference batches."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

@dataclass
class InferenceTask:
    task_id: str
    task_type: TaskType
    params: dict[str, Any]
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    status: TaskStatus = TaskStatus.PENDING
    progress_current: int = 0
    progress_total: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_info(self) -> TaskInfo:
        return TaskInfo(
            task_id=self.task_id,
            task_type=self.task_type,
            status=self.status,
            progress_current=self.progress_current,
            progress_total=self.progress_total,
            error=self.error,
        )
ProgressCallback = Callable[[str, int, int, dict], None]
CompletionCallback = Callable[[str, str, Optional[dict], Optional[str]], None]

class InferenceWorker:
    """Manages the ScreenShot model and a task queue."""

    def __init__(
        self,
        model_checkpoint: str,
        model_dir: str,
        device: str = "auto",
    ) -> None:
        self.model_checkpoint = model_checkpoint
        self.model_dir = model_dir
        self.device = device
        self._screenshot = None
        self._preprocessor = None
        self._drug_library = None
        self._queue: asyncio.Queue[InferenceTask] = asyncio.Queue()
        self._tasks: dict[str, InferenceTask] = {}
        self._progress_callbacks: list[ProgressCallback] = []
        self._completion_callbacks: list[CompletionCallback] = []
        self._worker_task: Optional[asyncio.Task] = None
        self._control_state: ControlState = ControlState()

    # ---- properties --------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._screenshot is not None

    @property
    def actual_device(self) -> str:
        if self._screenshot is not None:
            return str(self._screenshot.device)
        return "not_loaded"

    @property
    def device_name(self) -> str:
        if self._screenshot is not None:
            return str(self._screenshot.device)
        return self.device or "auto"

    # ---- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Load model and start the background processing loop."""
        await self._load_model()
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info("InferenceWorker started")

    async def stop(self) -> None:
        """Cancel all pending tasks and shut down."""
        for task in self._tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.cancel_token.cancel()
                task.status = TaskStatus.CANCELLED

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._screenshot = None
        self._drug_library = None
        self._preprocessor = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("InferenceWorker stopped")

    async def _load_model(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        from screenshot import ScreenShot, Preprocessor
        from datascreen import DrugLibrary

        logger.info("Loading drug library from %s ...", self.model_dir)
        self._drug_library = DrugLibrary.from_pretrained(self.model_dir)

        logger.info("Creating preprocessor ...")
        self._preprocessor = Preprocessor(self._drug_library, n_drugs=3)

        logger.info("Loading ScreenShot model from %s ...", self.model_checkpoint)
        self._screenshot = ScreenShot.from_pretrained(
            self.model_checkpoint, device=self.device
        )
        logger.info("Model loaded on %s", self._screenshot.device)

    def predict_combination(self, input_df, query_df, sample_id):
        input_sample = input_df[input_df["sample_id"] == sample_id]
        input_processed = self._preprocessor.transform(input_sample)
        query_processed = self._preprocessor.transform(query_df)

        result = self._screenshot.predict(
            df_input=input_processed,
            df_query=query_processed,
            n_drugs=2,
            sample_id_col="sample_id",
            col_viability="float_value",
            show_progress=False,
        )
        return result["predicted_viability"].tolist()

    # ---- task management ---------------------------------------------------

    def submit(self, task_type: TaskType, params: dict[str, Any]) -> str:
        """Submit a new inference task. Returns the task_id."""
        task_id = uuid.uuid4().hex[:12]
        task = InferenceTask(task_id=task_id, task_type=task_type, params=params)
        self._tasks[task_id] = task
        self._queue.put_nowait(task)
        logger.info("Task %s submitted (%s)", task_id, task_type)
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a task. Returns True if the task was found."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.cancel_token.cancel()
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
        return True

    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        return self._tasks.get(task_id)

    def on_progress(self, callback: ProgressCallback) -> None:
        self._progress_callbacks.append(callback)

    def on_completion(self, callback: CompletionCallback) -> None:
        self._completion_callbacks.append(callback)

    # ---- control management ------------------------------------------------

    def get_controls(self) -> dict:
        return self._control_state.get_config()

    def set_controls(self, sample_ids: list[str]) -> dict:
        return self._control_state.set_controls(sample_ids)

    # ---- delta computation -------------------------------------------------

    def compute_deltas(self, task_id: str) -> Optional[dict]:
        task = self.get_task(task_id)
        if task is None:
            return None
        if task.status != TaskStatus.COMPLETED:
            return "not_completed"

        predictions = task.result.get("predictions", []) if task.result else []
        return _compute_deltas_impl(predictions, self._control_state)

    # ---- background loop ---------------------------------------------------

    async def _process_loop(self) -> None:
        """Continuously pull tasks from the queue and execute them."""
        while True:
            task = await self._queue.get()
            if task.cancel_token.is_cancelled:
                task.status = TaskStatus.CANCELLED
                self._emit_completion(task.task_id, "cancelled", None, None)
                self._queue.task_done()
                continue

            task.status = TaskStatus.RUNNING
            try:
                await self._execute_task(task)
                if task.cancel_token.is_cancelled and task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.CANCELLED
                if task.status == TaskStatus.COMPLETED:
                    self._emit_completion(task.task_id, "completed", task.result, None)
                elif task.status == TaskStatus.CANCELLED:
                    self._emit_completion(task.task_id, "cancelled", None, None)
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                self._emit_completion(task.task_id, "cancelled", None, None)
                raise
            except Exception as exc:
                logger.exception("Task %s failed", task.task_id)
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                self._emit_completion(task.task_id, "failed", None, str(exc))
            finally:
                self._queue.task_done()

    async def _execute_task(self, task: InferenceTask) -> None:
        """Dispatch to the right handler based on task type."""
        loop = asyncio.get_event_loop()

        if task.task_type == TaskType.PREDICT:
            await loop.run_in_executor(None, self._run_predict, task)
        elif task.task_type == TaskType.PREDICT_FULL:
            await loop.run_in_executor(None, self._run_predict_full, task)

        if not task.cancel_token.is_cancelled and task.result is not None:
            task.status = TaskStatus.COMPLETED

    # ---- inference handlers (run in thread) --------------------------------

    def _emit_progress(self, task_id: str, current: int, total: int, data: dict) -> None:
        for cb in self._progress_callbacks:
            try:
                cb(task_id, current, total, data)
            except Exception:
                logger.exception("Progress callback error")

    def _emit_completion(self, task_id: str, status: str, result: Optional[dict], error: Optional[str]) -> None:
        callbacks = getattr(self, '_completion_callbacks', [])
        for cb in callbacks:
            try:
                cb(task_id, status, result, error)
            except Exception:
                logger.exception("Completion callback error")

    @staticmethod
    def _merge_float_value(preds_df, df_query_orig, sid, query_indices=None):
        if "float_value" not in df_query_orig.columns:
            return preds_df
        if query_indices is not None:
            orig_rows = df_query_orig.loc[query_indices]
        else:
            orig_rows = df_query_orig.loc[df_query_orig["sample_id"] == sid]
        if len(orig_rows) == len(preds_df):
            preds_df = preds_df.copy()
            preds_df["float_value"] = orig_rows["float_value"].values
        return preds_df

    @staticmethod
    def _merge_original_doses(preds_df, df_query_orig, sid, n_drugs: int = 3, query_indices=None):
        if query_indices is not None:
            orig_rows = df_query_orig.loc[query_indices]
        else:
            orig_rows = df_query_orig.loc[df_query_orig["sample_id"] == sid]
        if len(orig_rows) != len(preds_df):
            return preds_df
        preds_df = preds_df.copy()
        for i in range(1, n_drugs + 1):
            col = f"dose{i}"
            if col in orig_rows.columns and col in preds_df.columns:
                preds_df[col] = orig_rows[col].values
        return preds_df

    @staticmethod
    def _simplify_drug_names(preds_df, df_query_orig, sid, n_drugs: int = 3, query_indices=None):
        if query_indices is not None:
            orig_rows = df_query_orig.loc[query_indices]
        else:
            orig_rows = df_query_orig.loc[df_query_orig["sample_id"] == sid]
        if len(orig_rows) != len(preds_df):
            preds_df = preds_df.copy()
            for i in range(1, n_drugs + 1):
                col = f"drug{i}"
                if col in preds_df.columns:
                    preds_df[col] = preds_df[col].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if not isinstance(x, list) else "")
                    )
            return preds_df
        preds_df = preds_df.copy()
        for i in range(1, n_drugs + 1):
            col = f"drug{i}"
            if col in orig_rows.columns and col in preds_df.columns:
                preds_df[col] = orig_rows[col].values
        return preds_df

    def _run_predict(self, task: InferenceTask) -> None:
        """Run basic prediction, streaming results per sample."""
        import pandas as pd
        from io import StringIO
        from datascreen import extract_drug_names

        input_csv = task.params.get("input_csv", "")
        query_csv = task.params.get("query_csv") or input_csv
        n_drugs = task.params.get("n_drugs", 3)

        df_input = pd.read_csv(StringIO(input_csv))
        df_query = pd.read_csv(StringIO(query_csv))

        new_drugs = extract_drug_names(df_input)
        self._drug_library.update(new_drugs)
        df_input_proc = self._preprocessor.transform(df_input)
        df_query_proc = self._preprocessor.transform(df_query)

        sample_ids = df_query["sample_id"].unique().tolist()
        task.progress_total = len(sample_ids)

        all_results = []
        for i, sid in enumerate(sample_ids):
            if task.cancel_token.is_cancelled:
                return

            mask_input = df_input_proc["sample_id"] == sid
            mask_query = df_query_proc["sample_id"] == sid

            preds = self._screenshot.predict(
                df_input=df_input_proc[mask_input],
                df_query=df_query_proc[mask_query],
                n_drugs=n_drugs,
                show_progress=False,
            )
            preds = self._preprocessor.inverse_transform_drugs(preds)
            preds = self._simplify_drug_names(preds, df_query, sid, n_drugs)
            preds = self._merge_original_doses(preds, df_query, sid, n_drugs)
            preds = self._merge_float_value(preds, df_query, sid)

            task.progress_current = i + 1
            partial = preds.to_dict(orient="records")
            all_results.extend(partial)
            self._emit_progress(task.task_id, i + 1, len(sample_ids), {
                "sample_id": sid,
                "predictions": partial,
            })

        if not task.cancel_token.is_cancelled:
            metrics = compute_metrics_per_pair(all_results)
            task.result = {
                "predictions": all_results,
                "metrics": metrics,
            }

    def _run_predict_full(self, task: InferenceTask) -> None:
        """Run prediction per (sample, drug), streaming results progressively."""
        import pandas as pd
        from io import StringIO
        from datascreen import extract_drug_names

        input_csv = task.params.get("input_csv", "")
        query_csv = task.params.get("query_csv") or input_csv
        n_drugs = task.params.get("n_drugs", 3)

        df_input = pd.read_csv(StringIO(input_csv))
        df_query = pd.read_csv(StringIO(query_csv))

        new_drugs = extract_drug_names(df_input)
        self._drug_library.update(new_drugs)
        df_input_proc = self._preprocessor.transform(df_input)
        df_query_proc = self._preprocessor.transform(df_query)
        pairs = []
        for sid in df_query["sample_id"].unique():
            q_mask = df_query["sample_id"] == sid
            drugs = df_query.loc[q_mask, "drug1"].unique()
            for drug in drugs:
                pairs.append((sid, drug))

        task.progress_total = len(pairs)
        all_predictions = []
        input_cache = {}

        for i, (sid, drug) in enumerate(pairs):
            if task.cancel_token.is_cancelled:
                return

            if sid not in input_cache:
                input_cache[sid] = df_input_proc[df_input_proc["sample_id"] == sid]
            mask_query = (df_query_proc["sample_id"] == sid)
            q_sample = df_query_proc[mask_query]
            orig_mask = (df_query["sample_id"] == sid) & (df_query["drug1"] == drug)
            query_indices = df_query.index[orig_mask]
            q_drug = q_sample[q_sample.index.isin(query_indices)]

            if len(q_drug) == 0:
                continue

            preds = self._screenshot.predict(
                df_input=input_cache[sid],
                df_query=q_drug,
                n_drugs=n_drugs,
                show_progress=False,
            )
            preds = self._preprocessor.inverse_transform_drugs(preds)
            preds = self._simplify_drug_names(preds, df_query, sid, n_drugs, query_indices=query_indices)
            preds = self._merge_original_doses(preds, df_query, sid, n_drugs, query_indices=query_indices)
            preds = self._merge_float_value(preds, df_query, sid, query_indices=query_indices)

            task.progress_current = i + 1
            partial_preds = preds.to_dict(orient="records")
            all_predictions.extend(partial_preds)

            self._emit_progress(task.task_id, i + 1, len(pairs), {
                "sample_id": sid,
                "predictions": partial_preds,
            })

        if not task.cancel_token.is_cancelled:
            metrics = compute_metrics_per_pair(all_predictions)
            task.result = {
                "predictions": all_predictions,
                "metrics": metrics,
            }
