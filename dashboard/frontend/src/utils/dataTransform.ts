
import type {
  Prediction,
  SamplePredictionResult,
  InputDataPoint,
} from '../types/dose-response';

export function groupBySample(
  results: SamplePredictionResult[],
): Map<string, Prediction[]> {
  const map = new Map<string, Prediction[]>();
  for (const result of results) {
    const existing = map.get(result.sample_id) || [];
    existing.push(...result.predictions);
    map.set(result.sample_id, existing);
  }
  return map;
}

export function groupByDrug(
  predictions: Prediction[],
): Map<string, Prediction[]> {
  const map = new Map<string, Prediction[]>();
  for (const pred of predictions) {
    const key = getDrugKey(pred);
    const existing = map.get(key) || [];
    existing.push(pred);
    map.set(key, existing);
  }
  return map;
}

function normalizeDrugName(name: unknown): string {
  if (!name) return '';
  if (Array.isArray(name)) return String(name[0] || '');
  const s = String(name);
  // If comma-separated aliases, take the first
  const commaIdx = s.indexOf(',');
  return commaIdx >= 0 ? s.slice(0, commaIdx).trim() : s;
}

export function getDrugKey(
  pred: Pick<Prediction, 'drug1' | 'drug2' | 'drug3'>,
): string {
  const parts = [normalizeDrugName(pred.drug1)];
  if (pred.drug2) parts.push(normalizeDrugName(pred.drug2));
  if (pred.drug3) parts.push(normalizeDrugName(pred.drug3));
  return parts.filter(Boolean).sort().join(' + ');
}

export function getLogDose(dose: number): number {
  if (dose <= 0) {
    // Already in log scale or zero
    return dose;
  }
  return Math.log10(Math.max(dose, 1e-10));
}

export function getInputPointsForDrug(
  inputData: InputDataPoint[],
  sampleId: string,
  drugKey: string,
): InputDataPoint[] {
  const drugKeyParts = drugKey.split(' + ');
  return inputData.filter((point) => {
    if (point.sample_id !== sampleId) return false;
    const pointDrugKey = getDrugKey(point);
    if (pointDrugKey === drugKey) return true;
    // Handle alias-expanded names: each part may contain comma-separated aliases
    const pointParts = pointDrugKey.split(' + ');
    if (pointParts.length !== drugKeyParts.length) return false;
    return pointParts.every((inputDrug) =>
      drugKeyParts.some((predDrug) =>
        predDrug.split(',').some((alias) => alias.trim() === inputDrug.trim()),
      ),
    );
  });
}

export function getDrugLabel(drugKey: string): string {
  // drugKey is already normalized by getDrugKey, but handle any residual aliases
  return drugKey
    .split(' + ')
    .map((part) => {
      const commaIdx = part.indexOf(',');
      return commaIdx >= 0 ? part.slice(0, commaIdx).trim() : part.trim();
    })
    .join(' + ');
}

export function groupAllByDrug(
  results: SamplePredictionResult[],
): Map<string, { sampleId: string; predictions: Prediction[] }[]> {
  const map = new Map<
    string,
    { sampleId: string; predictions: Prediction[] }[]
  >();

  for (const result of results) {
    const drugGroups = groupByDrug(result.predictions);
    for (const [drugKey, preds] of drugGroups) {
      const existing = map.get(drugKey) || [];
      existing.push({ sampleId: result.sample_id, predictions: preds });
      map.set(drugKey, existing);
    }
  }

  return map;
}
