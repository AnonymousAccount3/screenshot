
export type TaskStatus = 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';

export type TaskType = 'predict' | 'predict_full';

export interface TaskInfo {
  task_id: string;
  task_type: TaskType;
  status: TaskStatus;
  progress_current: number;
  progress_total: number;
  error?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
}

export interface WSMessage {
  type: string;
  task_id: string;
  current?: number;
  total?: number;
  data?: Record<string, unknown>;
  detail?: string;
}

export interface InferenceRequest {
  task_type: TaskType;
  input_csv?: string;
  query_csv?: string;
  n_drugs?: number;
  sample_ids?: string[];
}

// Control sample configuration
export interface ControlConfig {
  method: 'user_selected' | 'median';
  control_sample_ids: string[];
}

// Delta computation result
export interface DeltaRow {
  sample_id: string;
  drug1: string;
  dose1: number;
  drug2?: string;
  dose2?: number;
  drug3?: string;
  dose3?: number;
  predicted_viability: number;
  control_viability: number;
  delta: number;
}

export interface DeltaResult {
  deltas: DeltaRow[];
  method: string;
  control_sample_ids: string[];
}

// Prediction row
export interface PredictionRow {
  sample_id: string;
  drug1: string;
  dose1: number;
  drug2?: string;
  dose2?: number;
  drug3?: string;
  dose3?: number;
  predicted_viability: number;
  float_value?: number;
}

// Metrics for a (sample_id, drug) pair
export interface DoseResponseMetrics {
  auc: number;
  ic50: number | null;
}

// Predict result
export interface PredictResult {
  predictions: PredictionRow[];
  metrics?: Record<string, DoseResponseMetrics>;  // keyed by "sample_id|drug1"
}

// Predict full result (predictions + embeddings)
export interface PredictFullResult extends PredictResult {
  sample_embeddings: number[][];
  drug_dose_embeddings?: number[][];
}

// C6: Drug toxicity lookup
export interface ToxicityReference {
  title: string;
  url: string;
}

export interface ToxicityInfo {
  name: string;
  toxicity_grade: number | null;
  common_toxicities?: string[];
  max_tolerated_dose?: string | null;
  black_box_warning?: boolean;
  references?: ToxicityReference[];
  message?: string;
}

// Analytics — delta summary per (sample, drug) pair
export interface DeltaSummary {
  sample_id: string;
  drug_key: string;
  avg_viability: number;      // mean predicted viability across doses
  control_viability: number;  // mean control viability across doses
  delta: number;              // mean of (control - predicted) per dose
}

// Ranked hit for bar plot
export interface RankedHit {
  sample_id: string;
  drug_key: string;
  delta: number;
}

// Analytics metric toggle
export type AnalyticsMetric = 'hits' | 'performance';

// MAE summary per (sample, drug) pair
export interface MAESummary {
  sample_id: string;
  drug_key: string;
  mae: number;
  n_points: number;
}
