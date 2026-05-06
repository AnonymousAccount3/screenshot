
import { useMemo } from 'react';
import type { ControlConfig, DeltaSummary, RankedHit, MAESummary, PredictionRow } from '../types/api';
import { flattenPartialResults, computeDeltaSummaries, rankHits, computeMAESummaries } from '../utils/analytics';

interface UseDeltaComputationReturn {
  deltas: DeltaSummary[];
  rankedHits: RankedHit[];
  maeSummaries: MAESummary[];
  flatPredictions: PredictionRow[];
}

export function useDeltaComputation(
  partialResults: Record<string, unknown>[],
  controlConfig: ControlConfig,
): UseDeltaComputationReturn {
  const flatPredictions = useMemo(() => {
    return flattenPartialResults(partialResults);
  }, [partialResults]);

  const deltas = useMemo(() => {
    return computeDeltaSummaries(flatPredictions, controlConfig);
  }, [flatPredictions, controlConfig]);

  const rankedHits = useMemo(() => {
    return rankHits(deltas);
  }, [deltas]);

  const maeSummaries = useMemo(() => {
    return computeMAESummaries(flatPredictions);
  }, [flatPredictions]);

  return { deltas, rankedHits, maeSummaries, flatPredictions };
}
