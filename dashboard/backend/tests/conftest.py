"""Shared fixtures for backend tests.

Uses a mock InferenceWorker so tests run without the real model.
"""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.cache import ResultCache
from app.controls import ControlState
from app.inference import InferenceWorker, InferenceTask, CancellationToken
from app.schemas import TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Mock drug library
# ---------------------------------------------------------------------------


class MockDrugLibrary:
    """Minimal drug library mock for testing Silo 1 endpoints."""

    def __init__(self):
        self.drug_id_remap = {}  # Empty dict: no drugs known by default
        self._canonical_names = {}

    def update(self, drug_names):
        """No-op for mock."""
        pass

    def get_canonical_name(self, drug_name):
        return self._canonical_names.get(drug_name)

    @property
    def all_drugs(self):
        import pandas as pd
        return pd.DataFrame(columns=["name", "pubchem_name", "smiles"])


# ---------------------------------------------------------------------------
# Mock worker
# ---------------------------------------------------------------------------


class MockInferenceWorker(InferenceWorker):
    """InferenceWorker that skips model loading and returns fake results."""

    def __init__(self):
        # Don't call super().__init__ — skip model paths
        self._screenshot = MagicMock()
        self._preprocessor = MagicMock()
        self._drug_library = MockDrugLibrary()
        self._queue = asyncio.Queue()
        self._tasks = {}
        self._progress_callbacks = []
        self._worker_task = None

        # Cache (D4)
        self._cache = ResultCache()

        # Control state (B3)
        self._control_state = ControlState()

        # Input state
        self._current_input_hash = None
        self._current_input_csv = None
        self._precomputed_heads = None
        self._viability_predictor_w = None
        self._viability_predictor_b = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._process_loop())

    def _load_model_sync(self):
        pass  # No-op

    @property
    def model_loaded(self):
        return True

    @property
    def actual_device(self):
        return "cpu_mock"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_worker():
    return MockInferenceWorker()


@pytest.fixture
async def app_client(mock_worker):
    """AsyncClient wired to the FastAPI app with a mock worker."""
    from app.main import app
    import app.main as main_module
    from app.session_store import session_store

    # Clear session store between tests
    session_store.clear()

    # Patch the global worker
    original_worker = main_module.worker
    main_module.worker = mock_worker
    await mock_worker.start()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await mock_worker.stop()
    main_module.worker = original_worker

    # Clear session store after test
    session_store.clear()
