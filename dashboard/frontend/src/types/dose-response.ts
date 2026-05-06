
import type { PredictionRow } from './api';

export type Prediction = PredictionRow;

export interface SamplePredictionResult {
  sample_id: string;
  predictions: Prediction[];
}

export interface InputDataPoint {
  sample_id: string;
  drug1: string;
  dose1: number;
  drug2?: string;
  dose2?: number;
  drug3?: string;
  dose3?: number;
  float_value?: number;
    _rowIndex: number;
}

export interface SampleCurve {
  sampleId: string;
  predictions: Prediction[];
  color: string;
}
