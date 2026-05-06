
import { memo, useMemo, useState, useEffect } from 'react';
import DoseResponseChart from './DoseResponseChart';
import { groupByDrug, getDrugKey } from '../utils/dataTransform';
import type { Prediction, InputDataPoint } from '../types/dose-response';

interface SampleSectionProps {
  sampleId: string;
  predictions: Prediction[];
  inputPoints: InputDataPoint[];
  subsampledIndices?: Set<number>;
  isCollapsed: boolean;
  onToggleCollapse: (sampleId: string) => void;
  columnsPerRow?: number;
  doseRange?: [number, number];
  onChartClick?: (drugKey: string, sampleId: string) => void;
  metrics?: Record<string, { auc: number; ic50: number | null }>;
}

const CHARTS_PER_FRAME = 3;

export default memo(function SampleSection({
  sampleId,
  predictions,
  inputPoints,
  subsampledIndices,
  isCollapsed,
  onToggleCollapse,
  columnsPerRow = 5,
  doseRange,
  onChartClick,
  metrics = {},
}: SampleSectionProps) {
  const drugGroups = useMemo(() => groupByDrug(predictions), [predictions]);
  const drugEntries = useMemo(() => Array.from(drugGroups.entries()), [drugGroups]);

  // Group input points by drug — only depends on inputPoints (static after upload)
  const inputByDrug = useMemo(() => {
    const map = new Map<string, InputDataPoint[]>();
    for (const p of inputPoints) {
      const key = getDrugKey(p);
      const arr = map.get(key);
      if (arr) arr.push(p);
      else map.set(key, [p]);
    }
    return map;
  }, [inputPoints]);

  const INITIAL_VISIBLE = 10;
  const LOAD_MORE_COUNT = 20;
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE);

  // Reset when collapsed
  useEffect(() => {
    if (isCollapsed) {
      setVisibleCount(INITIAL_VISIBLE);
    }
  }, [isCollapsed]);

  return (
    <div data-testid="sample-section" style={{ marginBottom: 'var(--space-4)' }}>
      <div
        data-testid="sample-section-header"
        onClick={() => onToggleCollapse(sampleId)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-2)',
          padding: 'var(--space-2) var(--space-3)',
          backgroundColor: 'var(--surface-muted)',
          borderRadius: 'var(--radius-md)',
          cursor: 'pointer',
          userSelect: 'none',
          fontWeight: 'var(--font-weight-semibold)',
          fontSize: 'var(--text-md)',
          border: '1px solid var(--border-subtle)',
          transition: 'background-color 150ms ease',
        }}
      >
        <span>{isCollapsed ? '\u25B8' : '\u25BE'}</span>
        <span>{sampleId}</span>
        <span style={{ color: 'var(--text-tertiary)', fontWeight: 'var(--font-weight-normal)', fontSize: 'var(--text-sm)' }}>
          ({drugGroups.size} drug{drugGroups.size !== 1 ? 's' : ''})
        </span>
      </div>

      {!isCollapsed && (
        <div
          data-testid="sample-section-content"
        >
          <div
            className="chart-grid"
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${columnsPerRow}, minmax(0, 1fr))`,
              gap: 'var(--space-6) var(--space-5)',
              padding: 'var(--space-4) 0',
            }}
          >
            {drugEntries.slice(0, visibleCount).map(([drugKey, preds]) => {
              const filteredInput = inputByDrug.get(drugKey) ?? [];
              const metricKey = `${sampleId}|${drugKey}`;
              const pairMetrics = metrics[metricKey];

              return (
                <div key={drugKey} onClick={() => onChartClick?.(drugKey, sampleId)} style={{ cursor: onChartClick ? 'pointer' : undefined }}>
                  <DoseResponseChart
                    drugLabel={drugKey}
                    sampleId={sampleId}
                    predictions={preds}
                    inputPoints={filteredInput}
                    subsampledIndices={subsampledIndices}
                    showQueryPoints={true}
                    doseRange={doseRange}
                    metrics={pairMetrics}
                  />
                </div>
              );
            })}
          </div>
          {visibleCount < drugEntries.length && (
            <button
              onClick={(e) => { e.stopPropagation(); setVisibleCount(v => Math.min(v + LOAD_MORE_COUNT, drugEntries.length)); }}
              style={{ margin: 'var(--space-3) 0', padding: '4px 12px', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}
            >
              Show more ({drugEntries.length - visibleCount} remaining)
            </button>
          )}
        </div>
      )}
    </div>
  );
})
