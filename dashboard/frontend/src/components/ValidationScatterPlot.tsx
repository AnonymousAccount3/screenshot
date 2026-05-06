
import { useMemo, useState, useCallback, useEffect } from 'react';
import { scaleLinear } from 'd3-scale';
import type { PredictionRow } from '../types/api';
import { getDrugKey } from '../utils/analytics';

interface ValidationScatterPlotProps {
  predictions: PredictionRow[];
}

interface ScatterPoint {
  actual: number;
  predicted: number;
  drugKey: string;
}

interface SampleStats {
  mae: number;
  r2: number;
  n: number;
}

const PLOT_WIDTH = 260;
const PLOT_HEIGHT = 260;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

function computeStats(points: ScatterPoint[]): SampleStats {
  if (points.length === 0) return { mae: 0, r2: 0, n: 0 };
  let sumAbsErr = 0;
  let sumActual = 0;
  for (const p of points) {
    sumAbsErr += Math.abs(p.predicted - p.actual);
    sumActual += p.actual;
  }
  const mae = sumAbsErr / points.length;
  const meanActual = sumActual / points.length;

  // R² = 1 - SS_res / SS_tot
  let ssRes = 0;
  let ssTot = 0;
  for (const p of points) {
    ssRes += (p.actual - p.predicted) ** 2;
    ssTot += (p.actual - meanActual) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  return { mae, r2, n: points.length };
}

function SampleScatter({
  sampleId,
  points,
  stats,
}: {
  sampleId: string;
  points: ScatterPoint[];
  stats: SampleStats;
}) {
  const innerW = PLOT_WIDTH - MARGIN.left - MARGIN.right;
  const innerH = PLOT_HEIGHT - MARGIN.top - MARGIN.bottom;

  const xScale = useMemo(() => scaleLinear().domain([0, 1.05]).range([0, innerW]), [innerW]);
  const yScale = useMemo(() => scaleLinear().domain([0, 1.05]).range([innerH, 0]), [innerH]);

  const svgWidth = PLOT_WIDTH;
  const svgHeight = PLOT_HEIGHT;

  return (
    <div className="chart-card">
      <h4 style={{ margin: '0 0 var(--space-2) 0', fontSize: 'var(--text-base)', color: 'var(--text-secondary)' }}>
        {sampleId}
      </h4>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto' }}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* Y=X reference line */}
          <line
            x1={xScale(0)} y1={yScale(0)}
            x2={xScale(1.05)} y2={yScale(1.05)}
            stroke="var(--color-neutral-300)"
            strokeWidth={1}
            strokeDasharray="4,3"
          />

          {/* Points */}
          {points.map((p, i) => (
            <circle
              key={i}
              cx={xScale(p.actual)}
              cy={yScale(p.predicted)}
              r={3}
              fill="var(--color-brand-500)"
              opacity={0.6}
            />
          ))}

          {/* X-axis */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="var(--color-neutral-200)" strokeWidth={1} />
          {xScale.ticks(5).map(tick => (
            <g key={`x-${tick}`}>
              <line x1={xScale(tick)} y1={innerH} x2={xScale(tick)} y2={innerH + 4} stroke="var(--color-neutral-400)" />
              <text x={xScale(tick)} y={innerH + 14} textAnchor="middle" fontSize={9} fill="var(--text-tertiary)">{tick.toFixed(1)}</text>
            </g>
          ))}

          {/* Y-axis */}
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="var(--color-neutral-200)" strokeWidth={1} />
          {yScale.ticks(5).map(tick => (
            <g key={`y-${tick}`}>
              <line x1={-4} y1={yScale(tick)} x2={0} y2={yScale(tick)} stroke="var(--color-neutral-400)" />
              <text x={-8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" fontSize={9} fill="var(--text-tertiary)">{tick.toFixed(1)}</text>
            </g>
          ))}

          {/* Axis labels */}
          <text x={innerW / 2} y={innerH + 32} textAnchor="middle" fontSize={10} fill="var(--text-secondary)">
            Actual Viability
          </text>
          <text transform="rotate(-90)" x={-innerH / 2} y={-36} textAnchor="middle" fontSize={10} fill="var(--text-secondary)">
            Predicted Viability
          </text>

          {/* Stats box */}
          <text x={innerW - 4} y={14} textAnchor="end" fontSize={9} fill="var(--text-tertiary)">
            MAE = {stats.mae.toFixed(4)}
          </text>
          <text x={innerW - 4} y={26} textAnchor="end" fontSize={9} fill="var(--text-tertiary)">
            R² = {stats.r2.toFixed(3)}
          </text>
          <text x={innerW - 4} y={38} textAnchor="end" fontSize={9} fill="var(--text-tertiary)">
            n = {stats.n}
          </text>
        </g>
      </svg>
    </div>
  );
}

export default function ValidationScatterPlot({ predictions }: ValidationScatterPlotProps) {
  const [zoomedSample, setZoomedSample] = useState<string | null>(null);
  const closeZoom = useCallback(() => setZoomedSample(null), []);

  useEffect(() => {
    if (!zoomedSample) return;
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeZoom(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [zoomedSample, closeZoom]);

  // Group predictions by sample, keeping only rows with float_value
  const sampleData = useMemo(() => {
    const map = new Map<string, ScatterPoint[]>();
    for (const row of predictions) {
      if (row.float_value === undefined || row.float_value === null) continue;
      const pts = map.get(row.sample_id);
      const point: ScatterPoint = {
        actual: row.float_value,
        predicted: row.predicted_viability,
        drugKey: getDrugKey(row),
      };
      if (pts) pts.push(point);
      else map.set(row.sample_id, [point]);
    }
    return map;
  }, [predictions]);

  // Stats per sample
  const sampleStats = useMemo(() => {
    const map = new Map<string, SampleStats>();
    for (const [sid, pts] of sampleData) {
      map.set(sid, computeStats(pts));
    }
    return map;
  }, [sampleData]);

  const sampleIds = useMemo(() => Array.from(sampleData.keys()).sort(), [sampleData]);

  const navigateSample = useCallback((direction: -1 | 1) => {
    if (!zoomedSample || sampleIds.length <= 1) return;
    const currentIdx = sampleIds.indexOf(zoomedSample);
    const len = sampleIds.length;
    const nextIdx = ((currentIdx + direction) % len + len) % len;
    setZoomedSample(sampleIds[nextIdx]);
  }, [zoomedSample, sampleIds]);

  useEffect(() => {
    if (!zoomedSample) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); navigateSample(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); navigateSample(1); }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [zoomedSample, navigateSample]);

  if (sampleIds.length === 0) return null;

  return (
    <div data-testid="validation-scatter">
      <h3 style={{ margin: '0 0 12px 0', fontSize: 'var(--text-lg)', fontWeight: 'var(--font-weight-semibold)' }}>
        Validation: Predicted vs Actual
      </h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
          gap: 'var(--space-3)',
        }}
      >
        {sampleIds.map(sid => (
          <div key={sid} onClick={() => setZoomedSample(sid)} style={{ cursor: 'pointer' }}>
            <SampleScatter
              sampleId={sid}
              points={sampleData.get(sid)!}
              stats={sampleStats.get(sid)!}
            />
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {zoomedSample && sampleData.get(zoomedSample) && (
        <div className="chart-lightbox-backdrop" onClick={closeZoom}>
          <div className="chart-lightbox-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
              <h4 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>{zoomedSample}</h4>
              <button onClick={closeZoom} style={{ background: 'none', border: 'none', fontSize: '1.5em', cursor: 'pointer', color: 'var(--text-tertiary)', padding: '4px' }}>&times;</button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              {sampleIds.length > 1 && (
                <button
                  onClick={() => navigateSample(-1)}
                  style={{ background: 'none', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '12px 6px', cursor: 'pointer', fontSize: 'var(--text-lg)', color: 'var(--text-secondary)', flexShrink: 0, lineHeight: 1 }}
                  title="Previous sample (←)"
                >◀</button>
              )}
              <div style={{ flex: 1, minWidth: 0 }}>
                <SampleScatter
                  sampleId={zoomedSample}
                  points={sampleData.get(zoomedSample)!}
                  stats={sampleStats.get(zoomedSample)!}
                />
              </div>
              {sampleIds.length > 1 && (
                <button
                  onClick={() => navigateSample(1)}
                  style={{ background: 'none', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '12px 6px', cursor: 'pointer', fontSize: 'var(--text-lg)', color: 'var(--text-secondary)', flexShrink: 0, lineHeight: 1 }}
                  title="Next sample (→)"
                >▶</button>
              )}
            </div>
            {sampleIds.length > 1 && (
              <div style={{ textAlign: 'center', marginTop: 'var(--space-2)', fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
                Sample {sampleIds.indexOf(zoomedSample) + 1} / {sampleIds.length}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
