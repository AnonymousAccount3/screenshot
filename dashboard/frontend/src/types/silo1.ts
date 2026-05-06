
export interface UploadResponse {
  row_count: number;
  columns: string[];
  column_mapping: Record<string, string>;
  degree: number;
  has_viability: boolean;
  dose_scale: 'linear' | 'log';
  dose_transform_applied: 'log10' | 'none';
  warnings: string[];
  sample_ids: string[];
  csv_text?: string;
}

export interface DrugInfo {
  name: string;
  known: boolean;
  canonical_name: string | null;
}

export interface DrugMappingResponse {
  drugs: DrugInfo[];
}

export interface QueryGenerateRequest {
  degree?: number;
  n_points?: number;
  dose_min?: number;
  dose_max?: number;
}

export interface QueryGenerateResponse {
  row_count: number;
  degree: number;
  preview: Record<string, unknown>[];
}

export interface CohortInfo {
  name: string;
  description: string;
  n_samples: number;
  n_drugs: number;
}

export interface CohortListResponse {
  cohorts: CohortInfo[];
}

export interface ColumnMappingOverride {
  sample_id?: string;
  drug1?: string;
  dose1?: string;
  drug2?: string;
  dose2?: string;
  drug3?: string;
  dose3?: string;
  viability?: string;
  group?: string;
}
