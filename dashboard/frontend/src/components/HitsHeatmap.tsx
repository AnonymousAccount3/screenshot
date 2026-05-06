
import { useState, useMemo, useCallback, useEffect } from 'react';
import { scaleSequential, scaleBand } from 'd3-scale';
import { interpolateRdBu, interpolateRdYlGn } from 'd3-scale-chromatic';
import type { DeltaSummary, MAESummary, AnalyticsMetric } from '../types/api';
import { getTopDrugKeys } from '../utils/analytics';

interface HitsHeatmapProps {
  deltas: DeltaSummary[];
  maeSummaries?: MAESummary[];
  topN: number;
  onTopNChange: (n: number) => void;
  onCellClick: (drugKey: string, sampleId: string) => void;
  metric?: AnalyticsMetric;
}

interface HoveredCell {
  sampleId: string;
  drugKey: string;
  x: number;
  y: number;
  // Hits mode
  delta?: number;
  controlViability?: number;
  avgViability?: number;
  // Performance mode
  mae?: number;
  nPoints?: number;
}

const LEFT_MARGIN = 160;
const TOP_MARGIN = 20;
const CELL_MIN_SIZE = 30;
const CELL_MAX_SIZE = 60;

function getTopDrugKeysByMAE(summaries: MAESummary[], topN?: number): string[] {
  const drugMap = new Map<string, number[]>();
  for (const s of summaries) {
    const arr = drugMap.get(s.drug_key) || [];
    arr.push(s.mae);
    drugMap.set(s.drug_key, arr);
  }
  const sorted = Array.from(drugMap.entries())
    .map(([drugKey, vals]) => ({ drugKey, meanMAE: vals.reduce((a, b) => a + b, 0) / vals.length }))
    .sort((a, b) => a.meanMAE - b.meanMAE);
  const limited = topN != null && topN > 0 ? sorted.slice(0, topN) : sorted;
  return limited.map(d => d.drugKey);
}

export default function HitsHeatmap({
  deltas,
  maeSummaries,
  topN,
  onTopNChange,
  onCellClick,
  metric = 'hits',
}: HitsHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null);
  const [zoomed, setZoomed] = useState(false);
  const closeZoom = useCallback(() => setZoomed(false), []);
  useEffect(() => {
    if (!zoomed) return;
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeZoom(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [zoomed, closeZoom]);

  const isPerf = metric === 'performance' && maeSummaries && maeSummaries.length > 0;

  // Drug keys: sorted by mean delta (hits) or mean MAE (performance)
  const drugKeys = useMemo(() => {
    if (isPerf) return getTopDrugKeysByMAE(maeSummaries!, topN);
    return getTopDrugKeys(deltas, topN);
  }, [isPerf, maeSummaries, deltas, topN]);

  // Sample IDs: sorted by mean delta desc (hits) or mean MAE asc (performance)
  const sampleIds = useMemo(() => {
    if (isPerf) {
      const sampleMap = new Map<string, number[]>();
      for (const s of maeSummaries!) {
        const arr = sampleMap.get(s.sample_id) || [];
        arr.push(s.mae);
        sampleMap.set(s.sample_id, arr);
      }
      return Array.from(sampleMap.entries())
        .map(([sid, vals]) => ({ sid, mean: vals.reduce((a, b) => a + b, 0) / vals.length }))
        .sort((a, b) => a.mean - b.mean)
        .map(e => e.sid);
    }
    const sampleDeltaMap = new Map<string, number[]>();
    for (const d of deltas) {
      const arr = sampleDeltaMap.get(d.sample_id) || [];
      arr.push(d.delta);
      sampleDeltaMap.set(d.sample_id, arr);
    }
    return Array.from(sampleDeltaMap.entries())
      .map(([sid, vals]) => ({ sid, mean: vals.reduce((a, b) => a + b, 0) / vals.length }))
      .sort((a, b) => b.mean - a.mean)
      .map(e => e.sid);
  }, [isPerf, maeSummaries, deltas]);

  // Cell lookup maps
  const deltaCellMap = useMemo(() => {
    const map = new Map<string, DeltaSummary>();
    for (const d of deltas) map.set(`${d.sample_id}||${d.drug_key}`, d);
    return map;
  }, [deltas]);

  const maeCellMap = useMemo(() => {
    if (!maeSummaries) return new Map<string, MAESummary>();
    const map = new Map<string, MAESummary>();
    for (const s of maeSummaries) map.set(`${s.sample_id}||${s.drug_key}`, s);
    return map;
  }, [maeSummaries]);

  // Color scales
  const maxDelta = useMemo(() => {
    if (deltas.length === 0) return 1;
    return Math.max(...deltas.map(d => Math.abs(d.delta)), 0.001);
  }, [deltas]);

  const maxMAE = useMemo(() => {
    if (!maeSummaries || maeSummaries.length === 0) return 0.2;
    return Math.max(...maeSummaries.map(s => s.mae), 0.001);
  }, [maeSummaries]);

  const colorScale = useMemo(() => {
    if (isPerf) {
      // Green (low MAE) → Red (high MAE): reverse RdYlGn so green=low
      return scaleSequential((t: number) => interpolateRdYlGn(1 - t)).domain([0, maxMAE]);
    }
    return scaleSequential(interpolateRdBu).domain([maxDelta, -maxDelta]);
  }, [isPerf, maxDelta, maxMAE]);

  // Layout
  const cellWidth = Math.max(CELL_MIN_SIZE, Math.min(CELL_MAX_SIZE, 500 / Math.max(drugKeys.length, 1)));
  const cellHeight = Math.max(CELL_MIN_SIZE, Math.min(CELL_MAX_SIZE, 400 / Math.max(sampleIds.length, 1)));
  const svgWidth = LEFT_MARGIN + drugKeys.length * cellWidth + 80;
  const svgHeight = TOP_MARGIN + sampleIds.length * cellHeight + 100;

  const xScale = useMemo(
    () => scaleBand<string>().domain(drugKeys).range([LEFT_MARGIN, LEFT_MARGIN + drugKeys.length * cellWidth]).padding(0.05),
    [drugKeys, cellWidth],
  );
  const yScale = useMemo(
    () => scaleBand<string>().domain(sampleIds).range([TOP_MARGIN, TOP_MARGIN + sampleIds.length * cellHeight]).padding(0.05),
    [sampleIds, cellHeight],
  );

  const handleMouseEnter = useCallback(
    (sid: string, drugKey: string, event: React.MouseEvent) => {
      const rect = (event.target as SVGRectElement).getBoundingClientRect();
      const base: HoveredCell = {
        sampleId: sid,
        drugKey,
        x: rect.left + rect.width / 2,
        y: rect.top - 10,
      };
      if (isPerf) {
        const s = maeCellMap.get(`${sid}||${drugKey}`);
        if (s) { base.mae = s.mae; base.nPoints = s.n_points; }
      } else {
        const d = deltaCellMap.get(`${sid}||${drugKey}`);
        if (d) { base.delta = d.delta; base.controlViability = d.control_viability; base.avgViability = d.avg_viability; }
      }
      setHoveredCell(base);
    },
    [isPerf, deltaCellMap, maeCellMap],
  );

  const handleMouseLeave = useCallback(() => setHoveredCell(null), []);

  // Get cell color for a given (sample, drug)
  const getCellColor = useCallback((sid: string, drugKey: string): string | null => {
    if (isPerf) {
      const s = maeCellMap.get(`${sid}||${drugKey}`);
      return s ? colorScale(s.mae) : null;
    }
    const d = deltaCellMap.get(`${sid}||${drugKey}`);
    return d ? colorScale(d.delta) : null;
  }, [isPerf, deltaCellMap, maeCellMap, colorScale]);

  const hasCell = useCallback((sid: string, drugKey: string): boolean => {
    if (isPerf) return maeCellMap.has(`${sid}||${drugKey}`);
    return deltaCellMap.has(`${sid}||${drugKey}`);
  }, [isPerf, deltaCellMap, maeCellMap]);

  if (deltas.length === 0 && (!maeSummaries || maeSummaries.length === 0)) return null;

  const title = isPerf ? 'MAE Heatmap' : 'Top Hits Heatmap';

  const renderSvg = (keyPrefix: string, onCellClickAction?: (drugKey: string, sampleId: string) => void) => (
    <svg width={svgWidth} height={svgHeight}>
      {sampleIds.map(sid =>
        drugKeys.map(drugKey => {
          if (!hasCell(sid, drugKey)) return null;
          const color = getCellColor(sid, drugKey);
          if (!color) return null;
          return (
            <rect
              key={`${keyPrefix}-${sid}-${drugKey}`}
              data-testid="heatmap-cell"
              x={xScale(drugKey)}
              y={yScale(sid)}
              width={xScale.bandwidth()}
              height={yScale.bandwidth()}
              fill={color}
              stroke="#fff"
              strokeWidth={1}
              style={{ cursor: 'pointer' }}
              onMouseEnter={(e) => handleMouseEnter(sid, drugKey, e)}
              onMouseLeave={handleMouseLeave}
              onClick={(e) => { e.stopPropagation(); setHoveredCell(null); onCellClickAction?.(drugKey, sid); }}
            />
          );
        }),
      )}
      {drugKeys.map(drugKey => {
        const labelY = TOP_MARGIN + sampleIds.length * cellHeight + 16;
        const labelX = (xScale(drugKey) ?? 0) + xScale.bandwidth() / 2;
        return (
          <text
            key={`${keyPrefix}-col-${drugKey}`}
            data-testid="heatmap-col-label"
            x={labelX}
            y={labelY}
            textAnchor="start"
            fontSize={10}
            fill="var(--text-secondary)"
            transform={`rotate(45, ${labelX}, ${labelY})`}
          >
            {drugKey.length > 25 ? drugKey.slice(0, 23) + '...' : drugKey}
          </text>
        );
      })}
      {sampleIds.map(sid => (
        <text
          key={`${keyPrefix}-row-${sid}`}
          data-testid="heatmap-row-label"
          x={LEFT_MARGIN - 12}
          y={(yScale(sid) ?? 0) + yScale.bandwidth() / 2}
          textAnchor="end"
          dominantBaseline="middle"
          fontSize={10}
          fill="var(--text-secondary)"
        >
          {sid.length > 16 ? sid.slice(0, 14) + '...' : sid}
        </text>
      ))}
    </svg>
  );

  return (
    <>
    <div data-testid="hits-heatmap" className="chart-card" style={{ position: 'relative', cursor: 'pointer', width: 'fit-content' }} onClick={() => setZoomed(true)}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
        <h3 style={{ margin: 0, fontSize: 'var(--text-base)' }}>{title}</h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.85em' }}>
          Top N:
          <input
            data-testid="top-n-filter"
            type="number"
            min={1}
            value={topN}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const val = parseInt(e.target.value, 10);
              if (!isNaN(val) && val > 0) onTopNChange(val);
            }}
            style={{ width: '60px', padding: '2px 6px' }}
          />
        </label>
      </div>

      <div style={{ overflowX: 'auto' }}>
        {renderSvg('main', onCellClick)}
      </div>

      {/* Tooltip */}
      {hoveredCell && (
        <div
          data-testid="heatmap-tooltip"
          style={{
            position: 'fixed',
            left: hoveredCell.x,
            top: hoveredCell.y,
            transform: 'translate(-50%, -100%)',
            backgroundColor: 'rgba(0,0,0,0.85)',
            color: '#fff',
            padding: '6px 10px',
            borderRadius: '4px',
            fontSize: '0.8em',
            pointerEvents: 'none',
            zIndex: 1100,
            whiteSpace: 'nowrap',
          }}
        >
          <div><strong>{hoveredCell.sampleId}</strong> / {hoveredCell.drugKey}</div>
          {hoveredCell.delta !== undefined && (
            <>
              <div>{'\u0394'} = {hoveredCell.delta.toFixed(3)}</div>
              <div>Avg Control: {hoveredCell.controlViability?.toFixed(3)}, Avg Predicted: {hoveredCell.avgViability?.toFixed(3)}</div>
            </>
          )}
          {hoveredCell.mae !== undefined && (
            <>
              <div>MAE = {hoveredCell.mae.toFixed(4)}</div>
              <div>n = {hoveredCell.nPoints} points</div>
            </>
          )}
        </div>
      )}
    </div>

    {/* Lightbox */}
    {zoomed && (
      <div className="chart-lightbox-backdrop" onClick={closeZoom}>
        <div className="chart-lightbox-content chart-lightbox-wide" onClick={(e) => e.stopPropagation()}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
            <h4 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>{title}</h4>
            <button onClick={closeZoom} style={{ background: 'none', border: 'none', fontSize: '1.5em', cursor: 'pointer', color: 'var(--text-tertiary)', padding: '4px' }}>&times;</button>
          </div>
          <div style={{ overflowX: 'auto' }}>
            {renderSvg('zoom', (dk, sid) => { onCellClick(dk, sid); closeZoom(); })}
          </div>
        </div>
      </div>
    )}
    </>
  );
}
