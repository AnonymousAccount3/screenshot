# ScreenShot Dashboard — Development Commands
# =============================================
# All commands agents need to develop and test the dashboard.

PYTHON = ../../venv/bin/python
PIP = ../../venv/bin/pip
UVICORN = ../../venv/bin/uvicorn

.PHONY: dev dev-backend dev-frontend test test-backend test-frontend test-e2e build install

# ---- Development -----------------------------------------------------------

dev: ## Start both backend and frontend in parallel
	npx concurrently "make dev-backend" "make dev-frontend"

dev-backend: ## Start FastAPI backend with auto-reload
	cd backend && $(PYTHON) -m app.main

dev-frontend: ## Start Vite dev server
	cd frontend && npm run dev

# ---- Testing ---------------------------------------------------------------

test: test-backend test-frontend test-e2e ## Run all tests

test-backend: ## Run backend pytest tests
	cd backend && $(PYTHON) -m pytest tests/ -v

test-frontend: ## Run frontend vitest tests
	cd frontend && npm test

test-e2e: ## Run Playwright E2E tests (headless)
	npx playwright test

test-e2e-headed: ## Run Playwright E2E tests with visible browser
	npx playwright test --headed

test-e2e-debug: ## Debug Playwright tests step by step
	npx playwright test --debug

# ---- Build -----------------------------------------------------------------

build: ## Build frontend for production
	cd frontend && npm run build

install: ## Install all dependencies
	$(PIP) install -e ../screenshot -e ../datascreen
	$(PIP) install fastapi "uvicorn[standard]" websockets pydantic python-multipart
	$(PIP) install pytest pytest-asyncio httpx
	cd frontend && npm install
	npm install
	npx playwright install chromium
