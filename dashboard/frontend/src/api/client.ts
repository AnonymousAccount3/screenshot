/** REST API client for the ScreenShot Dashboard backend. */

import type {
  ControlConfig,
  DeltaResult,
  HealthResponse,
  InferenceRequest,
  TaskInfo,
  ToxicityInfo,
} from '../types/api';
import type {
  UploadResponse,
  DrugMappingResponse,
  QueryGenerateRequest,
  QueryGenerateResponse,
  CohortListResponse,
  ColumnMappingOverride,
} from '../types/silo1';

const BASE_URL = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`API ${resp.status}: ${body}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  createTask: (req: InferenceRequest) =>
    request<TaskInfo>('/tasks', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  getTask: (taskId: string) => request<TaskInfo>(`/tasks/${taskId}`),

  cancelTask: (taskId: string) =>
    request<TaskInfo>(`/tasks/${taskId}/cancel`, { method: 'POST' }),

  getResult: (taskId: string) =>
    request<Record<string, unknown>>(`/tasks/${taskId}/result`),

  uploadTask: async (
    taskType: string,
    inputFile: File,
    queryFile?: File,
    nDrugs: number = 3,
  ): Promise<TaskInfo> => {
    const form = new FormData();
    form.append('task_type', taskType);
    form.append('n_drugs', String(nDrugs));
    form.append('input_file', inputFile);
    if (queryFile) form.append('query_file', queryFile);

    const resp = await fetch(`${BASE_URL}/tasks/upload`, {
      method: 'POST',
      body: form,
    });
    if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
    return resp.json() as Promise<TaskInfo>;
  },

  // --- Silo 1: Data Input ---

  uploadInput: async (csvText: string): Promise<UploadResponse> => {
    const resp = await fetch(`${BASE_URL}/upload/input`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/csv' },
      body: csvText,
    });
    if (!resp.ok) {
      const body = await resp.json();
      throw new Error(body.detail || `Upload failed: ${resp.status}`);
    }
    return resp.json();
  },

  uploadInputBinary: async (data: ArrayBuffer, contentType: string): Promise<UploadResponse> => {
    const resp = await fetch(`${BASE_URL}/upload/input`, {
      method: 'POST',
      headers: { 'Content-Type': contentType },
      body: data,
    });
    if (!resp.ok) {
      const body = await resp.json();
      throw new Error(body.detail || `Upload failed: ${resp.status}`);
    }
    return resp.json();
  },

  overrideMapping: (mapping: ColumnMappingOverride) =>
    request<UploadResponse>('/upload/input/mapping', {
      method: 'POST',
      body: JSON.stringify(mapping),
    }),

  getDrugMapping: () => request<DrugMappingResponse>('/drugs/mapping'),

  generateQuery: (params: QueryGenerateRequest) =>
    request<QueryGenerateResponse>('/query/generate', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  listCohorts: () => request<CohortListResponse>('/cohorts'),

  loadCohort: (name: string) =>
    request<UploadResponse>(`/cohorts/${name}/load`, { method: 'POST' }),

  // --- Session data (for inference submission) ---

  getSessionData: () =>
    request<{ input_csv: string; query_csv: string | null; n_samples: number }>('/session/data'),

  getSessionRestore: () =>
    request<{
      has_input: boolean;
      upload_response?: UploadResponse;
      input_csv?: string;
      has_query?: boolean;
      subsample_n?: number | null;
      dose_filter?: number[] | null;
    }>('/session/restore'),

  getSubsample: () =>
    request<{ n_per_sample: number | null; selected_indices: number[] | null }>('/session/subsample'),

  setSubsample: (nPerSample: number | null) =>
    request<{ n_per_sample: number | null; selected_indices: number[] | null }>('/session/subsample', {
      method: 'POST',
      body: JSON.stringify({ n_per_sample: nPerSample }),
    }),

  getDoseFilter: () =>
    request<{ selected_doses: number[] | null; available_doses: number[] | null; selected_indices: number[] | null }>('/session/dose-filter'),

  setDoseFilter: (selectedDoses: number[] | null) =>
    request<{ selected_doses: number[] | null; available_doses: number[] | null; selected_indices: number[] | null }>('/session/dose-filter', {
      method: 'POST',
      body: JSON.stringify({ selected_doses: selectedDoses }),
    }),

  // --- Silo 2: Inference Engine ---

  getControls: () => request<ControlConfig>('/controls'),

  setControls: (controlSampleIds: string[]) =>
    request<ControlConfig>('/controls', {
      method: 'POST',
      body: JSON.stringify({ control_sample_ids: controlSampleIds }),
    }),

  computeDeltas: (taskId: string) =>
    request<DeltaResult>('/compute/deltas', {
      method: 'POST',
      body: JSON.stringify({ task_id: taskId }),
    }),

  // --- Silo 5: Analytics (convenience endpoint) ---

  startInference: () =>
    request<TaskInfo>('/inference/start', { method: 'POST' }),

  // --- Silo 6: Toxicity lookup (C6) ---

  getToxicity: (drugName: string) =>
    request<ToxicityInfo>(`/toxicity/${encodeURIComponent(drugName)}`),
};
