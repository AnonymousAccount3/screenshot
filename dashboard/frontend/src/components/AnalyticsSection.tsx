
import { useState, useMemo } from 'react';
import type { DeltaSummary, RankedHit, ControlConfig, MAESummary, PredictionRow, AnalyticsMetric } from '../types/api';
import HitsHeatmap from './HitsHeatmap';
import HitsBarPlot from './HitsBarPlot';
import ValidationScatterPlot from './ValidationScatterPlot';
import GroupComparisonSection from './GroupComparisonSection';

function getControlLabel(config: ControlConfig): string {
  if (config.method === 'user_selected' && config.control_sample_ids.length > 0) {
    const ids = config.control_sample_ids;
    if (ids.length <= 3) return ids.join(', ');
    return `${ids.length} samples selected`;
  }
  return 'Median (all samples)';
}

const METRIC_LABELS: Record<AnalyticsMetric, string> = {
  hits: 'Top Hits',
  performance: 'Performance',
};

interface AnalyticsSectionProps {
  deltas: DeltaSummary[];
  rankedHits: RankedHit[];
  maeSummaries: MAESummary[];
  flatPredictions: PredictionRow[];
  controlConfig: ControlConfig;
  groupColumn: string | null;
  sampleGroups: Record<string, string>;
  onDrugClick: (drugKey: string, sampleId: string) => void;
    fullDataMAESummaries?: MAESummary[];
    fullDataFlatPredictions?: PredictionRow[];
    isComputingMAE?: boolean;
    maeReady?: boolean;
    maeProgress?: number;
    maeProgressCurrent?: number;
    maeProgressTotal?: number;
}

export default function AnalyticsSection({
  deltas,
  rankedHits,
  maeSummaries,
  flatPredictions,
  controlConfig,
  groupColumn,
  sampleGroups,
  onDrugClick,
  fullDataMAESummaries,
  fullDataFlatPredictions,
  isComputingMAE = false,
  maeReady = false,
  maeProgressCurrent = 0,
  maeProgressTotal = 0,
}: AnalyticsSectionProps) {
  const [topN, setTopN] = useState(20);
  const [metric, setMetric] = useState<AnalyticsMetric>('hits');

  // Check if performance mode is available:
  // Either from primary inference (query had float_value) or from full-data MAE run
  const hasFloatValues = useMemo(() => {
    return flatPredictions.some(p => p.float_value !== undefined && p.float_value !== null);
  }, [flatPredictions]);

  const performanceAvailable = hasFloatValues || maeReady;

  // Use full-data MAE results when available, fall back to primary inference MAE
  const effectiveMAESummaries = useMemo(() => {
    if (maeReady && fullDataMAESummaries && fullDataMAESummaries.length > 0) {
      return fullDataMAESummaries;
    }
    return maeSummaries;
  }, [maeReady, fullDataMAESummaries, maeSummaries]);

  const effectiveFlatPredictions = useMemo(() => {
    if (maeReady && fullDataFlatPredictions && fullDataFlatPredictions.length > 0) {
      return fullDataFlatPredictions;
    }
    return flatPredictions;
  }, [maeReady, fullDataFlatPredictions, flatPredictions]);

  // If no deltas, don't render anything
  if (deltas.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: 'var(--space-6)',
        padding: 'var(--space-5)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        backgroundColor: 'var(--surface-card)',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      {/* Header with metric toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-5)', flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0, fontSize: 'var(--text-xl)', fontWeight: 'var(--font-weight-semibold)' }}>
          Analytics
        </h2>

        {/* Metric toggle */}
        {performanceAvailable && (
          <div style={{ display: 'flex', gap: 'var(--space-1)' }}>
            {(['hits', 'performance'] as AnalyticsMetric[]).map((m) => (
              <button
                key={m}
                onClick={() => setMetric(m)}
                style={{
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: `1px solid ${metric === m ? 'var(--color-brand-300)' : 'var(--border-default)'}`,
                  backgroundColor: metric === m ? 'var(--color-brand-50)' : 'var(--surface-card)',
                  cursor: 'pointer',
                  fontSize: 'var(--text-sm)',
                  fontWeight: metric === m ? 'var(--font-weight-semibold)' : 'var(--font-weight-normal)',
                  color: metric === m ? 'var(--color-brand-700)' : 'var(--text-secondary)',
                  transition: 'all 150ms ease',
                }}
              >
                {METRIC_LABELS[m]}
              </button>
            ))}
          </div>
        )}

        {/* Background MAE computation indicator */}
        {isComputingMAE && !maeReady && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-2)',
            fontSize: 'var(--text-sm)',
            color: 'var(--text-tertiary)',
          }}>
            <span style={{
              display: 'inline-block',
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: 'var(--color-brand-400)',
              animation: 'pulse 1.5s ease-in-out infinite',
            }} />
            <span>
              Computing MAE on all data points...
              {maeProgressTotal > 0 && ` (${maeProgressCurrent}/${maeProgressTotal})`}
            </span>
            <style>{`@keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }`}</style>
          </div>
        )}

        {metric === 'hits' && (
          <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-tertiary)' }}>
            Control: {getControlLabel(controlConfig)}
          </span>
        )}
      </div>

      {/* Hits mode */}
      {metric === 'hits' && (
        <>
          {/* C3: Heatmap (delta) */}
          <div style={{ marginBottom: 'var(--space-6)' }}>
            <HitsHeatmap
              deltas={deltas}
              topN={topN}
              onTopNChange={setTopN}
              onCellClick={onDrugClick}
              metric="hits"
            />
          </div>

          {/* C5: Bar Plot */}
          <div style={{
            marginBottom: 'var(--space-6)',
            paddingTop: 'var(--space-5)',
            borderTop: '1px solid var(--border-subtle)',
          }}>
            <HitsBarPlot
              rankedHits={rankedHits}
              onBarClick={onDrugClick}
            />
          </div>
        </>
      )}

      {/* Performance mode */}
      {metric === 'performance' && (
        <>
          {/* Heatmap (MAE) */}
          <div style={{ marginBottom: 'var(--space-6)' }}>
            <HitsHeatmap
              deltas={deltas}
              maeSummaries={effectiveMAESummaries}
              topN={topN}
              onTopNChange={setTopN}
              onCellClick={onDrugClick}
              metric="performance"
            />
          </div>

          {/* Validation scatter plots */}
          <div style={{
            marginBottom: 'var(--space-6)',
            paddingTop: 'var(--space-5)',
            borderTop: '1px solid var(--border-subtle)',
          }}>
            <ValidationScatterPlot predictions={effectiveFlatPredictions} />
          </div>
        </>
      )}

      {/* C7: Group Comparisons (conditional, hits mode only) */}
      {metric === 'hits' && groupColumn && Object.keys(sampleGroups).length > 0 && (
        <div style={{
          paddingTop: 'var(--space-5)',
          borderTop: '1px solid var(--border-subtle)',
        }}>
          <GroupComparisonSection
            deltas={deltas}
            groups={sampleGroups}
            groupColumn={groupColumn}
          />
        </div>
      )}
    </div>
  );
}
