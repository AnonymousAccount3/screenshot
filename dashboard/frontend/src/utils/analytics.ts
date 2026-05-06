
import type { PredictionRow, DeltaSummary, RankedHit, ControlConfig, MAESummary } from '../types/api';

export function flattenPartialResults(
  partials: Record<string, unknown>[],
): PredictionRow[] {
  const rows: PredictionRow[] = [];
  for (const partial of partials) {
    const predictions = partial.predictions as PredictionRow[] | undefined;
    if (predictions && Array.isArray(predictions)) {
      rows.push(...predictions);
    }
  }
  return rows;
}

function normalizeDrug(name: unknown): string {
  if (!name) return '';
  if (Array.isArray(name)) return String(name[0] || '');
  const s = String(name);
  const commaIdx = s.indexOf(',');
  return commaIdx >= 0 ? s.slice(0, commaIdx).trim() : s;
}

export function getDrugKey(row: PredictionRow): string {
  const drugs: string[] = [normalizeDrug(row.drug1)];
  if (row.drug2) drugs.push(normalizeDrug(row.drug2));
  if (row.drug3) drugs.push(normalizeDrug(row.drug3));
  return drugs.filter(Boolean).sort().join(' + ');
}

function getDoseKey(row: PredictionRow): string {
  const parts: string[] = [String(row.dose1)];
  if (row.dose2 != null) parts.push(String(row.dose2));
  if (row.dose3 != null) parts.push(String(row.dose3));
  return parts.join('|');
}

export function computeDeltaSummaries(
  predictions: PredictionRow[],
  controlConfig: ControlConfig,
): DeltaSummary[] {
  if (predictions.length === 0) return [];

  // Index every prediction by (sample, drug, dose)
  const predMap = new Map<string, number>();
  const drugDoses = new Map<string, Set<string>>();
  const sampleDrugPairs = new Set<string>();
  const allSampleIds = new Set<string>();

  for (const row of predictions) {
    const drugKey = getDrugKey(row);
    const doseKey = getDoseKey(row);
    predMap.set(`${row.sample_id}||${drugKey}||${doseKey}`, row.predicted_viability);

    if (!drugDoses.has(drugKey)) drugDoses.set(drugKey, new Set());
    drugDoses.get(drugKey)!.add(doseKey);

    sampleDrugPairs.add(`${row.sample_id}||${drugKey}`);
    allSampleIds.add(row.sample_id);
  }

  // Compute control baseline per (drug, dose)
  const isUserSelected = controlConfig.method === 'user_selected'
    && controlConfig.control_sample_ids.length > 0;
  const controlSampleSet = new Set(controlConfig.control_sample_ids);
  const controlBaseline = new Map<string, number>();

  for (const [drugKey, doses] of drugDoses) {
    for (const doseKey of doses) {
      const values: number[] = [];
      for (const sid of allSampleIds) {
        if (isUserSelected && !controlSampleSet.has(sid)) continue;
        const v = predMap.get(`${sid}||${drugKey}||${doseKey}`);
        if (v !== undefined) values.push(v);
      }

      let baseline: number;
      if (isUserSelected) {
        baseline = values.length > 0
          ? values.reduce((a, b) => a + b, 0) / values.length
          : 0;
      } else {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        baseline = sorted.length % 2 === 0
          ? (sorted[mid - 1] + sorted[mid]) / 2
          : sorted[mid];
      }
      controlBaseline.set(`${drugKey}||${doseKey}`, baseline);
    }
  }

  // For each (sample, drug), compute average difference across all doses
  const summaries: DeltaSummary[] = [];
  for (const pairKey of sampleDrugPairs) {
    const sepIdx = pairKey.indexOf('||');
    const sampleId = pairKey.slice(0, sepIdx);
    const drugKey = pairKey.slice(sepIdx + 2);

    if (isUserSelected && controlSampleSet.has(sampleId)) continue;

    const doses = drugDoses.get(drugKey);
    if (!doses) continue;

    let sumDiff = 0;
    let sumViability = 0;
    let sumControl = 0;
    let count = 0;

    for (const doseKey of doses) {
      const predicted = predMap.get(`${sampleId}||${drugKey}||${doseKey}`);
      const control = controlBaseline.get(`${drugKey}||${doseKey}`);
      if (predicted !== undefined && control !== undefined) {
        sumDiff += control - predicted;
        sumViability += predicted;
        sumControl += control;
        count++;
      }
    }

    if (count > 0) {
      summaries.push({
        sample_id: sampleId,
        drug_key: drugKey,
        avg_viability: sumViability / count,
        control_viability: sumControl / count,
        delta: sumDiff / count,
      });
    }
  }

  return summaries;
}

export function rankHits(deltas: DeltaSummary[], topN?: number): RankedHit[] {
  const sorted = [...deltas].sort((a, b) => b.delta - a.delta);
  const limited = topN != null && topN > 0 ? sorted.slice(0, topN) : sorted;
  return limited.map((d) => ({
    sample_id: d.sample_id,
    drug_key: d.drug_key,
    delta: d.delta,
  }));
}

export function getTopDrugKeys(deltas: DeltaSummary[], topN?: number): string[] {
  // Compute mean delta per drug
  const drugDeltaMap = new Map<string, number[]>();
  for (const d of deltas) {
    const arr = drugDeltaMap.get(d.drug_key) || [];
    arr.push(d.delta);
    drugDeltaMap.set(d.drug_key, arr);
  }

  const drugMeanDeltas = Array.from(drugDeltaMap.entries()).map(([drugKey, vals]) => ({
    drugKey,
    meanDelta: vals.reduce((a, b) => a + b, 0) / vals.length,
  }));

  drugMeanDeltas.sort((a, b) => b.meanDelta - a.meanDelta);

  const limited = topN != null && topN > 0
    ? drugMeanDeltas.slice(0, topN)
    : drugMeanDeltas;

  return limited.map((d) => d.drugKey);
}

export function computeMAESummaries(predictions: PredictionRow[]): MAESummary[] {
  // Group by (sample_id, drug_key)
  const groups = new Map<string, { sumAbsErr: number; count: number }>();

  for (const row of predictions) {
    if (row.float_value === undefined || row.float_value === null) continue;
    const drugKey = getDrugKey(row);
    const key = `${row.sample_id}||${drugKey}`;
    const entry = groups.get(key);
    const absErr = Math.abs(row.predicted_viability - row.float_value);
    if (entry) {
      entry.sumAbsErr += absErr;
      entry.count += 1;
    } else {
      groups.set(key, { sumAbsErr: absErr, count: 1 });
    }
  }

  const summaries: MAESummary[] = [];
  for (const [key, { sumAbsErr, count }] of groups) {
    const sepIdx = key.indexOf('||');
    summaries.push({
      sample_id: key.slice(0, sepIdx),
      drug_key: key.slice(sepIdx + 2),
      mae: sumAbsErr / count,
      n_points: count,
    });
  }
  return summaries;
}

export function computeBlissScore(
  comboViability: number,
  singleAgentViabilities: number[],
): number {
  const expected = singleAgentViabilities.reduce((a, b) => a * b, 1);
  return comboViability - expected;
}

export function groupDeltasByGroup(
  deltas: DeltaSummary[],
  groups: Record<string, string>,
): Map<string, DeltaSummary[]> {
  const grouped = new Map<string, DeltaSummary[]>();
  for (const d of deltas) {
    const group = groups[d.sample_id] || 'Unknown';
    const arr = grouped.get(group) || [];
    arr.push(d);
    grouped.set(group, arr);
  }
  return grouped;
}

export function computeBoxPlotStats(values: number[]): {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
} {
  if (values.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0, mean: 0 };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mid = Math.floor(n / 2);
  const median = n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  const lowerHalf = sorted.slice(0, mid);
  const upperHalf = n % 2 === 0 ? sorted.slice(mid) : sorted.slice(mid + 1);

  const q1 = lowerHalf.length > 0
    ? lowerHalf[Math.floor(lowerHalf.length / 2)]
    : sorted[0];
  const q3 = upperHalf.length > 0
    ? upperHalf[Math.floor(upperHalf.length / 2)]
    : sorted[n - 1];

  return {
    min: sorted[0],
    q1,
    median,
    q3,
    max: sorted[n - 1],
    mean: sorted.reduce((a, b) => a + b, 0) / n,
  };
}
