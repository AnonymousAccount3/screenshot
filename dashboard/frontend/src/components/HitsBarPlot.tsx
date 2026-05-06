
import { useMemo, useState, useCallback, useEffect } from 'react';
import { scaleLinear, scaleBand } from 'd3-scale';
import type { RankedHit } from '../types/api';
import { useVivid } from '../contexts/VividContext';

interface HitsBarPlotProps {
  rankedHits: RankedHit[];
  onBarClick: (drugKey: string, sampleId: string) => void;
}

const PLOT_WIDTH = 300;
const PLOT_HEIGHT = 250;
const LEFT_MARGIN = 40;
const RIGHT_MARGIN = 80;
const TOP_MARGIN = 20;
const BOTTOM_MARGIN = 100;

// Single sample bar plot component
function SampleBarPlot({
  sampleId,
  hits,
  maxDelta,
  onBarClick,
  vivid,
}: {
  sampleId: string;
  hits: RankedHit[];
  maxDelta: number;
  onBarClick: (drugKey: string, sampleId: string) => void;
  vivid: boolean;
}) {
  const svgWidth = LEFT_MARGIN + PLOT_WIDTH + RIGHT_MARGIN;
  const svgHeight = TOP_MARGIN + PLOT_HEIGHT + BOTTOM_MARGIN;

  // Vertical bar chart: drugs on x-axis, delta on y-axis
  const yScale = useMemo(
    () => scaleLinear().domain([-maxDelta, maxDelta]).range([PLOT_HEIGHT, 0]),
    [maxDelta],
  );

  const xScale = useMemo(
    () =>
      scaleBand<string>()
        .domain(hits.map((h) => h.drug_key))
        .range([0, PLOT_WIDTH])
        .padding(0.2),
    [hits],
  );

  const zeroY = TOP_MARGIN + yScale(0);

  return (
    <div className="chart-card">
      <h4 style={{ margin: '0 0 var(--space-2) 0', fontSize: 'var(--text-base)', color: 'var(--text-secondary)' }}>
        {sampleId}
      </h4>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto' }}>
        {/* Zero line */}
        <line
          x1={LEFT_MARGIN}
          y1={zeroY}
          x2={LEFT_MARGIN + PLOT_WIDTH}
          y2={zeroY}
          stroke="var(--color-neutral-400)"
          strokeWidth={1}
          strokeDasharray="3,3"
        />

        {/* Bars */}
        {hits.map((hit) => {
          const xPos = xScale(hit.drug_key) ?? 0;
          const bandwidth = xScale.bandwidth();
          const barHeight = Math.abs(yScale(hit.delta) - yScale(0));
          const barY = hit.delta >= 0 ? zeroY - barHeight : zeroY;

          return (
            <g key={`${sampleId}-${hit.drug_key}`}>
              <rect
                data-testid="hit-bar"
                data-delta={hit.delta.toFixed(6)}
                x={LEFT_MARGIN + xPos}
                y={barY}
                width={bandwidth}
                height={barHeight}
                fill={hit.delta >= 0 ? (vivid ? '#b71c1c' : '#d32f2f') : (vivid ? '#0d47a1' : '#1565c0')}
                opacity={vivid ? 1 : 0.8}
                rx={2}
                style={{ cursor: 'pointer' }}
                onClick={(e) => { e.stopPropagation(); onBarClick(hit.drug_key, sampleId); }}
              />
            </g>
          );
        })}

        {/* Y-axis */}
        <line
          x1={LEFT_MARGIN}
          y1={TOP_MARGIN}
          x2={LEFT_MARGIN}
          y2={TOP_MARGIN + PLOT_HEIGHT}
          stroke="var(--color-neutral-200)"
          strokeWidth={1}
        />

        {/* Y-axis labels */}
        {yScale.ticks(4).map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={LEFT_MARGIN - 4}
              y1={TOP_MARGIN + yScale(tick)}
              x2={LEFT_MARGIN}
              y2={TOP_MARGIN + yScale(tick)}
              stroke="var(--color-neutral-400)"
            />
            <text
              x={LEFT_MARGIN - 8}
              y={TOP_MARGIN + yScale(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={8}
              fill="var(--text-tertiary)"
            >
              {Math.abs(tick) < 0.01 && tick !== 0 ? tick.toExponential(1) : tick.toFixed(3)}
            </text>
          </g>
        ))}

        {/* X-axis */}
        <line
          x1={LEFT_MARGIN}
          y1={TOP_MARGIN + PLOT_HEIGHT}
          x2={LEFT_MARGIN + PLOT_WIDTH}
          y2={TOP_MARGIN + PLOT_HEIGHT}
          stroke="var(--color-neutral-200)"
          strokeWidth={1}
        />

        {/* X-axis labels (drug names) */}
        {hits.map((hit) => {
          const xPos = xScale(hit.drug_key) ?? 0;
          const bandwidth = xScale.bandwidth();
          return (
            <text
              key={`x-${hit.drug_key}`}
              x={LEFT_MARGIN + xPos + bandwidth / 2}
              y={TOP_MARGIN + PLOT_HEIGHT + 8}
              textAnchor="start"
              fontSize={9}
              fill="var(--text-secondary)"
              transform={`rotate(45, ${LEFT_MARGIN + xPos + bandwidth / 2}, ${TOP_MARGIN + PLOT_HEIGHT + 8})`}
            >
              {hit.drug_key.length > 22 ? hit.drug_key.slice(0, 20) + '...' : hit.drug_key}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

const TOP_N_OPTIONS = [5, 10, 15, 20, 50];

export default function HitsBarPlot({
  rankedHits,
  onBarClick,
}: HitsBarPlotProps) {
  const vivid = useVivid();
  const [topN, setTopN] = useState(20);
  const [zoomedSample, setZoomedSample] = useState<string | null>(null);
  const closeZoom = useCallback(() => setZoomedSample(null), []);
  useEffect(() => {
    if (!zoomedSample) return;
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeZoom(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [zoomedSample, closeZoom]);

  // Group hits by sample_id (original logic, untouched)
  const sampleGroups = useMemo(() => {
    const groups = new Map<string, RankedHit[]>();
    for (const hit of rankedHits) {
      if (!groups.has(hit.sample_id)) {
        groups.set(hit.sample_id, []);
      }
      groups.get(hit.sample_id)!.push(hit);
    }
    return groups;
  }, [rankedHits]);

  // Calculate max delta for consistent scale across all samples (original logic)
  const maxDelta = useMemo(() => {
    if (rankedHits.length === 0) return 1;
    return Math.max(...rankedHits.map((h) => Math.abs(h.delta)), 0.001);
  }, [rankedHits]);

  // Display filter: top N per sample by delta (highest delta = top hits)
  const filteredGroups = useMemo(() => {
    const result = new Map<string, RankedHit[]>();
    for (const [sampleId, hits] of sampleGroups) {
      const sorted = [...hits].sort((a, b) => b.delta - a.delta);
      result.set(sampleId, sorted.slice(0, topN));
    }
    return result;
  }, [sampleGroups, topN]);

  const sampleIds = Array.from(filteredGroups.keys()).sort();

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

  if (sampleIds.length === 0) {
    return null;
  }

  return (
    <div data-testid="hits-barplot">
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', margin: '0 0 12px 0' }}>
        <h3 style={{ margin: 0, fontSize: 'var(--text-lg)', fontWeight: 'var(--font-weight-semibold)' }}>Top Hits by Sample</h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
          Show top
          <select
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            style={{ fontSize: 'var(--text-sm)', padding: '2px 8px' }}
          >
            {TOP_N_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          per sample
        </label>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
          gap: 'var(--space-3)',
        }}
      >
        {sampleIds.map((sampleId) => (
          <div key={sampleId} onClick={() => setZoomedSample(sampleId)} style={{ cursor: 'pointer' }}>
            <SampleBarPlot
              sampleId={sampleId}
              hits={filteredGroups.get(sampleId)!}
              maxDelta={maxDelta}
              onBarClick={onBarClick}
              vivid={vivid}
            />
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {zoomedSample && filteredGroups.get(zoomedSample) && (
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
                <SampleBarPlot
                  sampleId={zoomedSample}
                  hits={filteredGroups.get(zoomedSample)!}
                  maxDelta={maxDelta}
                  onBarClick={onBarClick}
                  vivid={vivid}
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
