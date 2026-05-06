
import { memo, useState, useMemo, useCallback, useEffect } from 'react';
import * as d3 from 'd3';
import SampleSection from './SampleSection';
import OverlayChart from './OverlayChart';
import GroupCompareChart from './GroupCompareChart';
import DoseResponseChart from './DoseResponseChart';
import ChartSkeleton from './ChartSkeleton';
import { groupByDrug, getDrugKey, getInputPointsForDrug } from '../utils/dataTransform';
import type {
  InputDataPoint,
  SampleCurve,
} from '../types/dose-response';
import type { ControlConfig } from '../types/api';
import './DoseResponseSection.css';

/* ── Filter picker (samples or drugs) ────────────────────────── */

function FilterPicker({
  label,
  allItems,
  selectedItems,
  onToggle,
  onSelectAll,
  onSelectNone,
}: {
  label: string;
  allItems: string[];
  selectedItems: Set<string>;
  onToggle: (item: string) => void;
  onSelectAll: () => void;
  onSelectNone: () => void;
}) {
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState(false);

  const filtered = useMemo(() => {
    if (!search) return allItems;
    const q = search.toLowerCase();
    return allItems.filter((item) => item.toLowerCase().includes(q));
  }, [allItems, search]);

  const selectedCount = selectedItems.size;
  const totalCount = allItems.length;

  return (
    <div style={{ minWidth: '180px', maxWidth: '280px' }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 10px',
          border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-sm)',
          cursor: 'pointer',
          fontSize: 'var(--text-sm)',
          backgroundColor: selectedCount < totalCount ? 'var(--color-warning-100)' : 'var(--surface-card)',
          transition: 'border-color 150ms ease',
        }}
      >
        <span>
          {label}: {selectedCount}/{totalCount}
        </span>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>{expanded ? '\u25B4' : '\u25BE'}</span>
      </div>

      {expanded && (
        <div
          style={{
            border: '1px solid var(--border-default)',
            borderTop: 'none',
            borderRadius: '0 0 var(--radius-sm) var(--radius-sm)',
            padding: '6px',
            backgroundColor: 'var(--surface-muted)',
            maxHeight: '220px',
            overflowY: 'auto',
          }}
        >
          <input
            type="text"
            placeholder={`Search ${label.toLowerCase()}...`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              marginBottom: '6px',
              boxSizing: 'border-box',
            }}
          />
          <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: '6px', fontSize: 'var(--text-xs)' }}>
            <button
              onClick={(e) => { e.stopPropagation(); onSelectAll(); }}
              style={{ background: 'none', border: 'none', color: 'var(--color-brand-600)', cursor: 'pointer', padding: 0 }}
            >
              Select all
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onSelectNone(); }}
              style={{ background: 'none', border: 'none', color: 'var(--color-error-600)', cursor: 'pointer', padding: 0 }}
            >
              Select none
            </button>
          </div>
          {filtered.map((item) => (
            <label
              key={item}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: 'var(--text-xs)',
                padding: '2px 0',
                cursor: 'pointer',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={selectedItems.has(item)}
                onChange={() => onToggle(item)}
                style={{ margin: 0 }}
              />
              <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {item}
              </span>
            </label>
          ))}
          {filtered.length === 0 && (
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', padding: '4px 0' }}>No matches</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── View mode ───────────────────────────────────────────────── */

type ViewMode = 'per-sample' | 'overlay' | 'group-compare';

const VIEW_MODE_LABELS: Record<ViewMode, string> = {
  'per-sample': 'Per-Sample',
  'overlay': 'Overlay',
  'group-compare': 'Group Compare',
};

const COLUMNS_OPTIONS = [2, 3, 4, 5, 6, 8];
const emptyInputPoints: InputDataPoint[] = [];

/* ── Main section ────────────────────────────────────────────── */

interface DoseResponseSectionProps {
  sampleMap: Map<string, Prediction[]>;
  inputData: InputDataPoint[];
  subsampledIndices?: Set<number> | null;
  isLoading: boolean;
  progressCurrent: number;
  progressTotal: number;
  controlConfig?: ControlConfig;
  metrics?: Record<string, { auc: number; ic50: number | null }>;
  externalZoom?: { drugKey: string; sampleId: string } | null;
  onExternalZoomClear?: () => void;
}

export default memo(function DoseResponseSection({
  sampleMap,
  inputData,
  subsampledIndices,
  isLoading,
  progressCurrent: _progressCurrent,
  progressTotal,
  controlConfig,
  metrics = {},
  externalZoom,
  onExternalZoomClear,
}: DoseResponseSectionProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [expandedSamples, setExpandedSamples] = useState<Set<string>>(new Set());
  // Track which views have been visited so we keep them mounted after first use
  const [mountedViews, setMountedViews] = useState<Set<ViewMode>>(new Set(['overlay']));
  const [columnsPerRow, setColumnsPerRow] = useState(5);

  // Dose range: applied values vs draft (user edits draft, applies on button/Enter)
  const [doseRange, setDoseRange] = useState<[number, number]>([-5, 3]);
  const [draftMin, setDraftMin] = useState('-5');
  const [draftMax, setDraftMax] = useState('3');
  const [doseRangeError, setDoseRangeError] = useState<string | null>(null);

  // Chart lightbox zoom
  const [zoomedChart, setZoomedChart] = useState<{ drugKey: string; mode: ViewMode; sampleId?: string } | null>(null);
  const closeZoom = useCallback(() => { setZoomedChart(null); onExternalZoomClear?.(); }, [onExternalZoomClear]);

  // Handle external zoom trigger from heatmap/barplot
  useEffect(() => {
    if (externalZoom) {
      setZoomedChart({ drugKey: externalZoom.drugKey, mode: 'per-sample', sampleId: externalZoom.sampleId });
    }
  }, [externalZoom]);
  useEffect(() => {
    if (!zoomedChart) return;
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeZoom(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [zoomedChart, closeZoom]);

  const applyDoseRange = useCallback(() => {
    const min = Number(draftMin);
    const max = Number(draftMax);
    if (isNaN(min) || isNaN(max)) {
      setDoseRangeError('Invalid number');
      return;
    }
    if (min >= max) {
      setDoseRangeError('Min must be less than max');
      return;
    }
    setDoseRangeError(null);
    setDoseRange([min, max]);
  }, [draftMin, draftMax]);

  const handleDoseRangeKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') applyDoseRange();
  }, [applyDoseRange]);

  const controlSampleIds = useMemo(
    () => new Set(controlConfig?.control_sample_ids ?? []),
    [controlConfig],
  );

  // All unique sample IDs and drug keys
  const allSampleIds = useMemo(
    () => Array.from(sampleMap.keys()).sort(),
    [sampleMap],
  );

  const allDrugKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const preds of sampleMap.values()) {
      for (const p of preds) {
        keys.add(getDrugKey(p));
      }
    }
    return Array.from(keys).sort();
  }, [sampleMap]);

  // Filter state — default: all selected
  const [selectedSamples, setSelectedSamples] = useState<Set<string>>(new Set());
  const [selectedDrugs, setSelectedDrugs] = useState<Set<string>>(new Set());

  // Sync defaults when predictions change (new inference)
  useMemo(() => {
    setSelectedSamples(new Set(allSampleIds));
    setSelectedDrugs(new Set(allDrugKeys));
  }, [allSampleIds, allDrugKeys]);

  const toggleSample = useCallback((id: string) => {
    setSelectedSamples((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  const toggleDrug = useCallback((key: string) => {
    setSelectedDrugs((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }, []);

  // Filtered sample map — stable refs for unchanged samples
  const sampleGroups = useMemo(() => {
    const filtered = new Map<string, Prediction[]>();
    for (const [sid, preds] of sampleMap) {
      if (!selectedSamples.has(sid)) continue;
      const filteredPreds = preds.filter((p) => selectedDrugs.has(getDrugKey(p)));
      if (filteredPreds.length > 0) {
        filtered.set(sid, filteredPreds);
      }
    }
    return filtered;
  }, [sampleMap, selectedSamples, selectedDrugs]);

  const handleViewModeChange = useCallback((mode: ViewMode) => {
    if (mode === 'per-sample') {
      const allKeys = Array.from(sampleGroups.keys());
      setExpandedSamples(new Set([allKeys[0]]));
    }
    setMountedViews((prev) => {
      if (prev.has(mode)) return prev;
      const next = new Set(prev);
      next.add(mode);
      return next;
    });
    setViewMode(mode);
  }, [sampleGroups]);

  // Group by drug across all samples (for overlay and group-compare views)
  const drugByAllSamples = useMemo(() => {
    const map = new Map<string, { sampleId: string; predictions: Prediction[] }[]>();
    for (const [sid, preds] of sampleGroups) {
      const byDrug = groupByDrug(preds);
      for (const [drugKey, drugPreds] of byDrug) {
        const existing = map.get(drugKey) || [];
        existing.push({ sampleId: sid, predictions: drugPreds });
        map.set(drugKey, existing);
      }
    }
    return map;
  }, [sampleGroups]);

  const drugOverlayGroups = useMemo(() => {
    const colors = d3.schemeCategory10;
    const sampleColorMap = new Map(
      allSampleIds.map((sid, i) => [sid, colors[i % colors.length]]),
    );
    const result = new Map<string, SampleCurve[]>();
    for (const [drugKey, sampleEntries] of drugByAllSamples) {
      result.set(drugKey, sampleEntries.map((entry) => ({
        sampleId: entry.sampleId,
        predictions: entry.predictions,
        color: sampleColorMap.get(entry.sampleId) || colors[0],
      })));
    }
    return result;
  }, [drugByAllSamples, allSampleIds]);

  const groupCompareData = useMemo(() => {
    const result = new Map<string, { sampleId: string; predictions: Prediction[]; isControl: boolean }[]>();
    for (const [drugKey, sampleEntries] of drugByAllSamples) {
      result.set(drugKey, sampleEntries.map((entry) => ({
        sampleId: entry.sampleId,
        predictions: entry.predictions,
        isControl: controlSampleIds.has(entry.sampleId),
      })));
    }
    return result;
  }, [drugByAllSamples, controlSampleIds]);

  const toggleCollapse = useCallback((sampleId: string) => {
    setExpandedSamples((prev) => {
      const next = new Set(prev);
      if (next.has(sampleId)) {
        next.delete(sampleId);
      } else {
        next.add(sampleId);
      }
      return next;
    });
  }, []);

  // Lightbox: samples that have data for the zoomed drug
  const zoomedDrugSamples = useMemo(() => {
    if (!zoomedChart) return [];
    return [...sampleGroups.entries()]
      .filter(([, preds]) => preds.some(p => getDrugKey(p) === zoomedChart.drugKey))
      .map(([sid]) => sid);
  }, [zoomedChart, sampleGroups]);

  const navigateSample = useCallback((direction: -1 | 1) => {
    if (!zoomedChart || zoomedChart.mode !== 'per-sample') return;
    const currentIdx = zoomedDrugSamples.indexOf(zoomedChart.sampleId || '');
    const len = zoomedDrugSamples.length;
    if (len === 0) return;
    const nextIdx = ((currentIdx + direction) % len + len) % len;
    setZoomedChart({ ...zoomedChart, sampleId: zoomedDrugSamples[nextIdx] });
  }, [zoomedChart, zoomedDrugSamples]);

  // Lightbox: all drug keys available for left/right navigation
  const zoomedDrugKeys = useMemo(() => {
    if (!zoomedChart) return [];
    // Use the drug keys from the current mode's data source
    if (zoomedChart.mode === 'overlay') return [...drugOverlayGroups.keys()];
    if (zoomedChart.mode === 'group-compare') return [...groupCompareData.keys()];
    // per-sample: drugs available for the current sample
    if (zoomedChart.sampleId) {
      const preds = sampleGroups.get(zoomedChart.sampleId);
      if (preds) {
        const keys = new Set(preds.map(p => getDrugKey(p)));
        return [...keys].sort();
      }
    }
    return [];
  }, [zoomedChart, drugOverlayGroups, groupCompareData, sampleGroups]);

  const navigateDrug = useCallback((direction: -1 | 1) => {
    if (!zoomedChart || zoomedDrugKeys.length <= 1) return;
    const currentIdx = zoomedDrugKeys.indexOf(zoomedChart.drugKey);
    const len = zoomedDrugKeys.length;
    const nextIdx = ((currentIdx + direction) % len + len) % len;
    setZoomedChart({ ...zoomedChart, drugKey: zoomedDrugKeys[nextIdx] });
  }, [zoomedChart, zoomedDrugKeys]);

  const handleLightboxModeChange = useCallback((newMode: ViewMode) => {
    if (!zoomedChart) return;
    if (newMode === 'per-sample') {
      const sampleId = zoomedChart.sampleId || zoomedDrugSamples[0];
      if (sampleId) {
        setZoomedChart({ drugKey: zoomedChart.drugKey, mode: newMode, sampleId });
      }
    } else {
      setZoomedChart({ drugKey: zoomedChart.drugKey, mode: newMode, sampleId: zoomedChart.sampleId });
    }
  }, [zoomedChart, zoomedDrugSamples]);

  // Arrow key navigation in lightbox
  useEffect(() => {
    if (!zoomedChart) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); navigateDrug(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); navigateDrug(1); }
      if (zoomedChart.mode === 'per-sample' && e.key === 'ArrowUp') { e.preventDefault(); navigateSample(-1); }
      if (zoomedChart.mode === 'per-sample' && e.key === 'ArrowDown') { e.preventDefault(); navigateSample(1); }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [zoomedChart, navigateSample, navigateDrug]);

  // Pre-group inputData by sample so each SampleSection gets only its own rows
  const inputBySample = useMemo(() => {
    const map = new Map<string, InputDataPoint[]>();
    for (const p of inputData) {
      const arr = map.get(p.sample_id);
      if (arr) arr.push(p);
      else map.set(p.sample_id, [p]);
    }
    return map;
  }, [inputData]);

  const remainingCount = 0;

  if (sampleMap.size === 0 && !isLoading) {
    return null;
  }

  return (
    <div data-testid="dose-response-section" className="dose-response-section">
      <div className="section-header">
        <h3 style={{ margin: 0 }}>Dose-Response Curves</h3>
        <div style={{ display: 'flex', gap: 'var(--space-1)' }}>
          {(['per-sample', 'overlay', 'group-compare'] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              data-testid={mode === 'overlay' ? 'overlay-mode-toggle' : undefined}
              onClick={() => handleViewModeChange(mode)}
              style={{
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${viewMode === mode ? 'var(--color-brand-300)' : 'var(--border-default)'}`,
                backgroundColor: viewMode === mode ? 'var(--color-brand-50)' : 'var(--surface-card)',
                cursor: 'pointer',
                fontSize: 'var(--text-sm)',
                fontWeight: viewMode === mode ? 'var(--font-weight-semibold)' : 'var(--font-weight-normal)',
                color: viewMode === mode ? 'var(--color-brand-700)' : 'var(--text-secondary)',
                transition: 'all 150ms ease',
              }}
            >
              {VIEW_MODE_LABELS[mode]}
            </button>
          ))}
        </div>
      </div>

      {/* Filter bar + display controls */}
      {sampleMap.size > 0 && (
        <div style={{ display: 'flex', gap: 'var(--space-3)', marginBottom: 'var(--space-3)', flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <FilterPicker
            label="Samples"
            allItems={allSampleIds}
            selectedItems={selectedSamples}
            onToggle={toggleSample}
            onSelectAll={() => setSelectedSamples(new Set(allSampleIds))}
            onSelectNone={() => setSelectedSamples(new Set())}
          />
          <FilterPicker
            label="Drugs"
            allItems={allDrugKeys}
            selectedItems={selectedDrugs}
            onToggle={toggleDrug}
            onSelectAll={() => setSelectedDrugs(new Set(allDrugKeys))}
            onSelectNone={() => setSelectedDrugs(new Set())}
          />

          {/* Columns per row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--text-sm)', whiteSpace: 'nowrap' }}>
            <label>Columns:</label>
            <select
              value={columnsPerRow}
              onChange={(e) => setColumnsPerRow(Number(e.target.value))}
              style={{ fontSize: 'var(--text-sm)', padding: '4px 8px' }}
            >
              {COLUMNS_OPTIONS.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

          {/* Log dose range (draft → apply) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--text-sm)', whiteSpace: 'nowrap' }}>
            <label>Dose range:</label>
            <input
              type="text"
              value={draftMin}
              onChange={(e) => setDraftMin(e.target.value)}
              onKeyDown={handleDoseRangeKeyDown}
              style={{
                width: '50px',
                borderColor: doseRangeError ? 'var(--color-error-600)' : undefined,
              }}
            />
            <span>to</span>
            <input
              type="text"
              value={draftMax}
              onChange={(e) => setDraftMax(e.target.value)}
              onKeyDown={handleDoseRangeKeyDown}
              style={{
                width: '50px',
                borderColor: doseRangeError ? 'var(--color-error-600)' : undefined,
              }}
            />
            <button
              onClick={applyDoseRange}
              style={{ padding: '4px 10px' }}
            >
              Apply
            </button>
            {doseRangeError && (
              <span style={{ color: 'var(--color-error-600)', fontSize: 'var(--text-xs)' }}>{doseRangeError}</span>
            )}
          </div>
        </div>
      )}

      <div style={{ display: viewMode === 'per-sample' ? 'block' : 'none' }}>
        {viewMode === 'per-sample' && (
          <>
            {Array.from(sampleGroups.entries()).map(([sampleId, preds]) => (
              <SampleSection
                key={sampleId}
                sampleId={sampleId}
                predictions={preds}
                inputPoints={inputBySample.get(sampleId) ?? emptyInputPoints}
                subsampledIndices={subsampledIndices ?? undefined}
                isCollapsed={!expandedSamples.has(sampleId)}
                onToggleCollapse={toggleCollapse}
                columnsPerRow={columnsPerRow}
                doseRange={doseRange}
                onChartClick={(drugKey, sid) => setZoomedChart({ drugKey, mode: 'per-sample', sampleId: sid })}
                metrics={metrics}
              />
            ))}
          </>
        )}
      </div>

      <div
        style={{
          display: viewMode === 'overlay' ? 'grid' : 'none',
          gridTemplateColumns: `repeat(${columnsPerRow}, minmax(0, 1fr))`,
          gap: 'var(--space-6) var(--space-5)',
          padding: 'var(--space-4) 0',
        }}
      >
        {(viewMode === 'overlay' || (!isLoading && mountedViews.has('overlay'))) &&
          Array.from(drugOverlayGroups.entries()).map(([drugKey, curves]) => (
            <div key={drugKey} onClick={() => setZoomedChart({ drugKey, mode: 'overlay' })} style={{ cursor: 'pointer' }}>
              <OverlayChart
                drugLabel={drugKey}
                sampleCurves={curves}
                doseRange={doseRange}
              />
            </div>
          ))
        }
      </div>

      <div style={{ display: viewMode === 'group-compare' ? 'block' : 'none' }}>
        {(viewMode === 'group-compare' || (!isLoading && mountedViews.has('group-compare'))) && (
          <>
            {/* Legend */}
            <div style={{ display: 'flex', gap: 'var(--space-4)', padding: 'var(--space-2) 0', fontSize: 'var(--text-sm)' }}>
              {controlSampleIds.size > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                  <div style={{ width: 24, height: 3, backgroundColor: 'var(--color-brand-700)', borderRadius: 1 }} />
                  <span>Control ({controlSampleIds.size})</span>
                </div>
              )}
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <div style={{ width: 24, height: 3, backgroundColor: 'var(--color-error-700)', borderRadius: 1 }} />
                <span>{controlSampleIds.size > 0 ? `Non-control (${allSampleIds.length - controlSampleIds.size})` : `All samples (${allSampleIds.length})`}</span>
              </div>
              <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--text-sm)' }}>Bold = median, transparent = individual</span>
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${columnsPerRow}, minmax(0, 1fr))`,
                gap: 'var(--space-3)',
                padding: 'var(--space-3) 0',
              }}
            >
              {Array.from(groupCompareData.entries()).map(([drugKey, curves]) => (
                <div key={drugKey} onClick={() => setZoomedChart({ drugKey, mode: 'group-compare' })} style={{ cursor: 'pointer' }}>
                  <GroupCompareChart
                    drugLabel={drugKey}
                    curves={curves}
                    doseRange={doseRange}
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Chart lightbox */}
      {zoomedChart && (
        <div className="chart-lightbox-backdrop" onClick={closeZoom}>
          <div className="chart-lightbox-content" onClick={(e) => e.stopPropagation()}>
            {/* Title row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-2)' }}>
              <h4 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>
                {zoomedChart.drugKey}
                {zoomedChart.mode === 'per-sample' && zoomedChart.sampleId && (
                  <span style={{ color: 'var(--text-secondary)', fontWeight: 'var(--font-weight-normal)' }}> — {zoomedChart.sampleId}</span>
                )}
              </h4>
              <button onClick={closeZoom} style={{ background: 'none', border: 'none', fontSize: '1.5em', cursor: 'pointer', color: 'var(--text-tertiary)', padding: '4px' }}>&times;</button>
            </div>

            {/* Mode toggle + sample navigation */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-3)', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', gap: 'var(--space-1)' }}>
                {(['per-sample', 'overlay', 'group-compare'] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => handleLightboxModeChange(mode)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: 'var(--radius-sm)',
                      border: `1px solid ${zoomedChart.mode === mode ? 'var(--color-brand-300)' : 'var(--border-default)'}`,
                      backgroundColor: zoomedChart.mode === mode ? 'var(--color-brand-50)' : 'var(--surface-card)',
                      cursor: 'pointer',
                      fontSize: 'var(--text-xs)',
                      fontWeight: zoomedChart.mode === mode ? 'var(--font-weight-semibold)' : 'var(--font-weight-normal)',
                      color: zoomedChart.mode === mode ? 'var(--color-brand-700)' : 'var(--text-secondary)',
                      transition: 'all 150ms ease',
                    }}
                  >
                    {VIEW_MODE_LABELS[mode]}
                  </button>
                ))}
              </div>
              {zoomedChart.mode === 'per-sample' && zoomedDrugSamples.length > 1 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--text-sm)' }}>
                  <button
                    onClick={() => navigateSample(-1)}
                    style={{ background: 'none', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '2px 8px', cursor: 'pointer', fontSize: 'var(--text-sm)' }}
                    title="Previous sample (↑)"
                  >
                    ▲
                  </button>
                  <span style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                    {zoomedDrugSamples.indexOf(zoomedChart.sampleId || '') + 1} / {zoomedDrugSamples.length}
                  </span>
                  <button
                    onClick={() => navigateSample(1)}
                    style={{ background: 'none', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '2px 8px', cursor: 'pointer', fontSize: 'var(--text-sm)' }}
                    title="Next sample (↓)"
                  >
                    ▼
                  </button>
                </div>
              )}
            </div>

            {/* Chart content with left/right drug navigation arrows */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              {zoomedDrugKeys.length > 1 && (
                <button
                  onClick={() => navigateDrug(-1)}
                  style={{
                    background: 'none',
                    border: '1px solid var(--border-default)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '12px 6px',
                    cursor: 'pointer',
                    fontSize: 'var(--text-lg)',
                    color: 'var(--text-secondary)',
                    flexShrink: 0,
                    lineHeight: 1,
                  }}
                  title="Previous drug (←)"
                >
                  ◀
                </button>
              )}
              <div style={{ flex: 1, minWidth: 0 }}>
                {zoomedChart.mode === 'overlay' && drugOverlayGroups.get(zoomedChart.drugKey) && (
                  <OverlayChart
                    drugLabel={zoomedChart.drugKey}
                    sampleCurves={drugOverlayGroups.get(zoomedChart.drugKey)!}
                    doseRange={doseRange}
                  />
                )}
                {zoomedChart.mode === 'group-compare' && groupCompareData.get(zoomedChart.drugKey) && (
                  <GroupCompareChart
                    drugLabel={zoomedChart.drugKey}
                    curves={groupCompareData.get(zoomedChart.drugKey)!}
                    doseRange={doseRange}
                  />
                )}
                {zoomedChart.mode === 'per-sample' && zoomedChart.sampleId && (() => {
                  const preds = sampleGroups.get(zoomedChart.sampleId!);
                  if (!preds) return null;
                  const drugPreds = preds.filter(p => getDrugKey(p) === zoomedChart.drugKey);
                  const filteredInput = getInputPointsForDrug(inputData, zoomedChart.sampleId!, zoomedChart.drugKey);
                  const metricKey = `${zoomedChart.sampleId}|${zoomedChart.drugKey}`;
                  const pairMetrics = metrics[metricKey];
                  return (
                    <DoseResponseChart
                      drugLabel={zoomedChart.drugKey}
                      sampleId={zoomedChart.sampleId!}
                      predictions={drugPreds}
                      inputPoints={filteredInput}
                      subsampledIndices={subsampledIndices ?? undefined}
                      showQueryPoints={true}
                      doseRange={doseRange}
                      metrics={pairMetrics}
                    />
                  );
                })()}
              </div>
              {zoomedDrugKeys.length > 1 && (
                <button
                  onClick={() => navigateDrug(1)}
                  style={{
                    background: 'none',
                    border: '1px solid var(--border-default)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '12px 6px',
                    cursor: 'pointer',
                    fontSize: 'var(--text-lg)',
                    color: 'var(--text-secondary)',
                    flexShrink: 0,
                    lineHeight: 1,
                  }}
                  title="Next drug (→)"
                >
                  ▶
                </button>
              )}
            </div>
            {zoomedDrugKeys.length > 1 && (
              <div style={{ textAlign: 'center', marginTop: 'var(--space-2)', fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
                Drug {zoomedDrugKeys.indexOf(zoomedChart.drugKey) + 1} / {zoomedDrugKeys.length}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
})
