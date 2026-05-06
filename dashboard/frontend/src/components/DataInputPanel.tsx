import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type {
  UploadResponse,
  DrugInfo,
  QueryGenerateResponse,
  ColumnMappingOverride,
} from '../types/silo1';
import type { InputDataPoint } from '../types/dose-response';
import type { ControlConfig } from '../types/api';
import { api } from '../api/client';
import UploadZone from './UploadZone';
import ColumnMappingPanel from './ColumnMappingPanel';
import DrugLibraryPanel from './DrugLibraryPanel';
import QueryGenerationPanel from './QueryGenerationPanel';
import CohortSelector from './CohortSelector';

// Dataset preview component with inline column header filters
const PAGE_SIZE = 50;

function DatasetPreview({ csvText }: { csvText: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [page, setPage] = useState(0);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const filterRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setPage(0);
    setSortCol(null);
    setSortAsc(true);
    setFilters({});
    setActiveFilter(null);
  }, [csvText]);

  // Close filter dropdown on outside click
  useEffect(() => {
    if (!activeFilter) return;
    const handleClick = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) {
        setActiveFilter(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [activeFilter]);

  const { headers, allRows } = useMemo(() => {
    const lines = csvText.trim().split('\n');
    const h = lines[0].split(',').map((v) => v.trim());
    const r = lines.slice(1).map((line) => line.split(',').map((v) => v.trim()));
    return { headers: h, allRows: r };
  }, [csvText]);

  const totalRows = allRows.length;

  const filteredRows = useMemo(() => {
    return allRows.filter((row) => {
      for (const [colHeader, filterText] of Object.entries(filters)) {
        if (!filterText) continue;
        const colIdx = headers.indexOf(colHeader);
        if (colIdx === -1) continue;
        if (!row[colIdx].toLowerCase().includes(filterText.toLowerCase())) {
          return false;
        }
      }
      return true;
    });
  }, [allRows, filters, headers]);

  const sortedRows = useMemo(() => {
    if (!sortCol) return filteredRows;
    const colIdx = headers.indexOf(sortCol);
    if (colIdx === -1) return filteredRows;
    const sorted = [...filteredRows].sort((a, b) => {
      const aVal = a[colIdx];
      const bVal = b[colIdx];
      const aNum = parseFloat(aVal);
      const bNum = parseFloat(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return sortAsc ? aNum - bNum : bNum - aNum;
      }
      return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    return sorted;
  }, [filteredRows, sortCol, sortAsc, headers]);

  const totalPages = Math.ceil(sortedRows.length / PAGE_SIZE);
  const displayedRows = sortedRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const hasActiveFilters = Object.values(filters).some((f) => f);

  const handleSortClick = (colHeader: string) => {
    if (sortCol === colHeader) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(colHeader);
      setSortAsc(true);
    }
    setPage(0);
  };

  return (
    <div className="card" style={{ marginTop: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: 'var(--text-base)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-2)',
          }}
        >
          <span>{isExpanded ? '\u25BE' : '\u25B8'}</span>
          Dataset Preview
          <span style={{ fontWeight: 'var(--font-weight-normal)', color: 'var(--text-tertiary)', fontSize: 'var(--text-sm)' }}>
            {sortedRows.length}/{totalRows} rows
          </span>
        </button>
        {hasActiveFilters && (
          <button
            onClick={() => setFilters({})}
            style={{
              padding: '2px 8px',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-error-600)',
              border: 'none',
              background: 'none',
            }}
          >
            Clear filters
          </button>
        )}
      </div>

      {isExpanded && (
        <div style={{ marginTop: 'var(--space-3)' }}>
          <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '400px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
            <table style={{ width: '100%' }}>
              <thead style={{ position: 'sticky', top: 0, zIndex: 5 }}>
                <tr>
                  {headers.map((h) => {
                    const isFiltered = !!filters[h];
                    const isSorted = sortCol === h;
                    return (
                      <th
                        key={h}
                        style={{
                          position: 'relative',
                          backgroundColor: isSorted ? 'var(--color-brand-50)' : isFiltered ? 'var(--color-brand-50)' : undefined,
                          padding: 'var(--space-2) var(--space-3)',
                          whiteSpace: 'nowrap',
                          userSelect: 'none',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <span
                            onClick={() => handleSortClick(h)}
                            style={{ cursor: 'pointer', flex: 1 }}
                          >
                            {h}
                            {isSorted && (
                              <span style={{ marginLeft: '3px', fontSize: '0.7em', opacity: 0.6 }}>
                                {sortAsc ? '\u25B2' : '\u25BC'}
                              </span>
                            )}
                          </span>
                          <span
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveFilter(activeFilter === h ? null : h);
                            }}
                            style={{
                              cursor: 'pointer',
                              fontSize: '10px',
                              opacity: isFiltered ? 1 : 0.35,
                              color: isFiltered ? 'var(--color-brand-600)' : 'var(--text-tertiary)',
                              padding: '2px',
                            }}
                            title={`Filter ${h}`}
                          >
                            &#9698;
                          </span>
                        </div>
                        {activeFilter === h && (
                          <div
                            ref={filterRef}
                            style={{
                              position: 'absolute',
                              top: '100%',
                              left: 0,
                              right: 0,
                              minWidth: '140px',
                              backgroundColor: 'var(--surface-elevated)',
                              border: '1px solid var(--border-default)',
                              borderRadius: '0 0 var(--radius-sm) var(--radius-sm)',
                              padding: '6px',
                              zIndex: 20,
                              boxShadow: 'var(--shadow-md)',
                            }}
                          >
                            <input
                              type="text"
                              placeholder={`Filter...`}
                              value={filters[h] || ''}
                              onChange={(e) => setFilters({ ...filters, [h]: e.target.value })}
                              onKeyDown={(e) => { if (e.key === 'Escape') setActiveFilter(null); }}
                              style={{
                                width: '100%',
                                boxSizing: 'border-box',
                                fontSize: 'var(--text-xs)',
                                padding: '4px 6px',
                              }}
                              autoFocus
                            />
                          </div>
                        )}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {displayedRows.length === 0 ? (
                  <tr>
                    <td colSpan={headers.length} style={{ textAlign: 'center', color: 'var(--text-tertiary)' }}>
                      No rows match filters
                    </td>
                  </tr>
                ) : (
                  displayedRows.map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j}>{cell}</td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div style={{ marginTop: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}
                style={{ padding: '2px 8px', fontSize: 'var(--text-sm)' }}>&laquo; Prev</button>
              <span>Page {page + 1} / {totalPages}</span>
              <button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}
                style={{ padding: '2px 8px', fontSize: 'var(--text-sm)' }}>Next &raquo;</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface DataInputPanelProps {
  onDataChange?: (hasInput: boolean, hasQuery: boolean, inputRows?: InputDataPoint[]) => void;
  onInputReady?: (response: UploadResponse) => void;
  onDrugClick?: (drugName: string) => void;
  onSubsampleIndicesChange?: (indices: Set<number> | null) => void;
  controlConfig: ControlConfig;
  onControlChange: (sampleIds: string[]) => void;
}

export default function DataInputPanel({ onDataChange, onInputReady, onDrugClick, onSubsampleIndicesChange, controlConfig, onControlChange }: DataInputPanelProps) {
  // Input upload state
  const [inputUpload, setInputUpload] = useState<UploadResponse | null>(null);
  const [inputFileName, setInputFileName] = useState<string | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);

  // Drug library state
  const [drugs, setDrugs] = useState<DrugInfo[]>([]);
  const [isDrugLoading, setIsDrugLoading] = useState(false);

  // Query generation state
  const [_queryGeneration, setQueryGeneration] = useState<QueryGenerateResponse | null>(null);

  // Raw input data for dose-response overlay
  const [inputCsvText, setInputCsvText] = useState<string | null>(null);

  // Track generated query state
  const [hasGeneratedQuery, setHasGeneratedQuery] = useState(false);

  // Subsample filter: N random rows per sample for inference
  const [subsampleN, setSubsampleN] = useState<number | null>(500);

  // Dose filter: select specific doses
  const [availableDoses, setAvailableDoses] = useState<number[]>([]);
  const [selectedDoses, setSelectedDoses] = useState<number[] | null>(null);

  const availableSampleIds = useMemo(() => {
    if (!inputUpload) return [];
    return inputUpload.sample_ids || [];
  }, [inputUpload]);

  // Per-sample row counts (for showing context in the filter UI)
  const sampleRowCounts = useMemo(() => {
    if (!inputCsvText || !inputUpload) return null;
    const sampleCol = inputUpload.column_mapping.sample_id;
    if (!sampleCol) return null;
    const lines = inputCsvText.trim().split('\n');
    if (lines.length < 2) return null;
    const headers = lines[0].split(',').map((h) => h.trim());
    const colIdx = headers.indexOf(sampleCol);
    if (colIdx === -1) return null;
    const counts = new Map<string, number>();
    for (let i = 1; i < lines.length; i++) {
      const val = lines[i].split(',')[colIdx]?.trim();
      if (val) counts.set(val, (counts.get(val) || 0) + 1);
    }
    const vals = [...counts.values()];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const avg = Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
    return { min, max, avg, nSamples: counts.size };
  }, [inputCsvText, inputUpload]);

  // Sync subsample setting to backend and propagate selected indices
  const handleSubsampleChange = useCallback(async (n: number | null) => {
    setSubsampleN(n);
    try {
      const resp = await api.setSubsample(n);
      onSubsampleIndicesChange?.(
        resp.selected_indices ? new Set(resp.selected_indices) : null,
      );
    } catch {
      onSubsampleIndicesChange?.(null);
    }
  }, [onSubsampleIndicesChange]);

  // Fetch available doses after upload
  const fetchAvailableDoses = useCallback(async () => {
    try {
      const resp = await api.getDoseFilter();
      if (resp.available_doses) setAvailableDoses(resp.available_doses);
      if (resp.selected_doses) setSelectedDoses(resp.selected_doses);
    } catch {
      setAvailableDoses([]);
    }
  }, []);

  // Handle dose filter change
  const handleDoseFilterChange = useCallback(async (doses: number[] | null) => {
    setSelectedDoses(doses);
    try {
      const resp = await api.setDoseFilter(doses);
      onSubsampleIndicesChange?.(
        resp.selected_indices ? new Set(resp.selected_indices) : null,
      );
    } catch {
      onSubsampleIndicesChange?.(null);
    }
  }, [onSubsampleIndicesChange]);

  // Fetch drug mapping after upload
  const fetchDrugMapping = useCallback(async () => {
    setIsDrugLoading(true);
    try {
      const resp = await api.getDrugMapping();
      setDrugs(resp.drugs);
    } catch {
      setDrugs([]);
    } finally {
      setIsDrugLoading(false);
    }
  }, []);

  // Parse CSV text into InputDataPoint array
  const parseCsvToInputData = useCallback((csvText: string): InputDataPoint[] => {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map((h) => h.trim());
    const rows: InputDataPoint[] = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map((v) => v.trim());
      const row: Record<string, string> = {};
      headers.forEach((h, idx) => {
        row[h] = values[idx] || '';
      });
      rows.push({
        sample_id: row.sample_id || '',
        drug1: row.drug1 || '',
        dose1: parseFloat(row.dose1) || 0,
        drug2: row.drug2 || undefined,
        dose2: row.dose2 ? parseFloat(row.dose2) : undefined,
        drug3: row.drug3 || undefined,
        dose3: row.dose3 ? parseFloat(row.dose3) : undefined,
        float_value: row.float_value ? parseFloat(row.float_value) : undefined,
        _rowIndex: i - 1,
      });
    }
    return rows;
  }, []);

  // Input upload handlers
  const handleInputSuccess = useCallback(
    (response: UploadResponse, fileName: string, csvText?: string) => {
      setInputUpload(response);
      setInputFileName(fileName);
      setInputError(null);
      setQueryGeneration(null);
      setHasGeneratedQuery(false);
      setInputCsvText(csvText || response.csv_text || null);
      fetchDrugMapping();
      fetchAvailableDoses();
      onInputReady?.(response);
    },
    [fetchDrugMapping, fetchAvailableDoses, onInputReady],
  );

  const handleInputError = useCallback((error: string) => {
    setInputError(error);
  }, []);

  // Column mapping override
  const handleMappingChange = useCallback(async (newMapping: ColumnMappingOverride) => {
    try {
      const response = await api.overrideMapping(newMapping);
      setInputUpload(response);
    } catch {
      // Silently fail mapping override
    }
  }, []);

  // Query generation handler
  const handleQueryGenerated = useCallback((response: QueryGenerateResponse) => {
    setQueryGeneration(response);
    setHasGeneratedQuery(true);
  }, []);

  // Cohort load handler
  const handleCohortLoaded = useCallback(
    (response: UploadResponse) => {
      setInputUpload(response);
      setInputFileName('cohort');
      setInputError(null);
      fetchDrugMapping();
      fetchAvailableDoses();
      onInputReady?.(response);
    },
    [fetchDrugMapping, fetchAvailableDoses, onInputReady],
  );

  // Restore session on mount (page refresh)
  useEffect(() => {
    let cancelled = false;
    async function restore() {
      try {
        const session = await api.getSessionRestore();
        if (cancelled || !session.has_input || !session.upload_response) return;
        setInputUpload(session.upload_response);
        setInputFileName('(restored)');
        if (session.input_csv) {
          setInputCsvText(session.input_csv);
        }
        if (session.has_query) {
          setHasGeneratedQuery(true);
        }
        if (session.subsample_n != null) {
          setSubsampleN(session.subsample_n);
          // Fetch selected indices for the restored subsample
          try {
            const sub = await api.getSubsample();
            onSubsampleIndicesChange?.(
              sub.selected_indices ? new Set(sub.selected_indices) : null,
            );
          } catch { /* ignore */ }
        }
        if (session.dose_filter != null) {
          setSelectedDoses(session.dose_filter);
        }
        fetchDrugMapping();
        fetchAvailableDoses();
        onInputReady?.(session.upload_response);
      } catch {
        // Backend not ready or no session — ignore
      }
    }
    restore();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Notify parent of data changes
  useEffect(() => {
    if (!onDataChange) return;
    const hasInputData = inputUpload !== null;
    const hasQueryData = hasGeneratedQuery;
    const inputRows = inputCsvText ? parseCsvToInputData(inputCsvText) : [];
    onDataChange(hasInputData, hasQueryData, inputRows);
  }, [inputUpload, hasGeneratedQuery, inputCsvText, onDataChange, parseCsvToInputData]);

  return (
    <div style={{ padding: 'var(--space-5)', width: '100%' }}>
      <h2 className="section-title">Data Input</h2>

      {/* Input upload zone */}
      <UploadZone
        type="input"
        onUploadSuccess={handleInputSuccess}
        onUploadError={handleInputError}
        uploadResponse={inputUpload}
        fileName={inputFileName}
        error={inputError}
      />

      {/* Dataset preview (shown after input upload) */}
      {inputUpload && inputCsvText && <DatasetPreview csvText={inputCsvText} />}

      {/* Input filters (shown after input upload) */}
      {inputUpload && sampleRowCounts && (
        <div className="card" style={{ marginTop: 'var(--space-3)' }}>
          <div style={{ fontSize: 'var(--text-sm)', fontWeight: 'var(--font-weight-semibold)', marginBottom: 'var(--space-2)' }}>
            Input Filters
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
            <label style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
              Rows per sample:
            </label>
            <select
              value={subsampleN ?? ''}
              onChange={(e) => {
                const val = e.target.value;
                handleSubsampleChange(val === '' ? null : Number(val));
              }}
              style={{ fontSize: 'var(--text-sm)', padding: '4px 8px' }}
            >
              <option value="">All ({sampleRowCounts.avg} avg)</option>
              {[50, 100, 200, 500, 1000, 2000].filter((n) => n < sampleRowCounts.max).map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

          {/* Dose filter */}
          {availableDoses.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-3)', flexWrap: 'wrap', marginTop: 'var(--space-3)', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--border-subtle)' }}>
              <label style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', whiteSpace: 'nowrap', paddingTop: '2px' }}>
                Dose filter:
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-1)' }}>
                {availableDoses.map((dose) => {
                  const isSelected = selectedDoses === null || selectedDoses.includes(dose);
                  return (
                    <button
                      key={dose}
                      onClick={() => {
                        if (selectedDoses === null) {
                          // First click: select only this dose
                          handleDoseFilterChange([dose]);
                        } else if (selectedDoses.includes(dose)) {
                          const remaining = selectedDoses.filter((d) => d !== dose);
                          handleDoseFilterChange(remaining.length === 0 ? null : remaining);
                        } else {
                          handleDoseFilterChange([...selectedDoses, dose]);
                        }
                      }}
                      style={{
                        padding: '2px 8px',
                        fontSize: 'var(--text-xs)',
                        borderRadius: 'var(--radius-pill)',
                        border: `1px solid ${isSelected ? 'var(--color-brand-500)' : 'var(--border-default)'}`,
                        background: isSelected ? 'var(--color-brand-50)' : 'var(--surface-card)',
                        color: isSelected ? 'var(--color-brand-700)' : 'var(--text-tertiary)',
                        cursor: 'pointer',
                        fontWeight: isSelected ? 'var(--font-weight-medium)' : 'var(--font-weight-normal)',
                      }}
                    >
                      {dose}
                    </button>
                  );
                })}
              </div>
              {selectedDoses !== null && (
                <button
                  onClick={() => handleDoseFilterChange(null)}
                  style={{
                    padding: '2px 8px',
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-error-600)',
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                  }}
                >
                  Reset
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Cohort selector (shown when no input uploaded) */}
      {!inputUpload && <CohortSelector onCohortLoaded={handleCohortLoaded} />}

      {/* Column mapping (shown after input upload) */}
      {inputUpload && (
        <ColumnMappingPanel
          uploadResponse={inputUpload}
          onMappingChange={handleMappingChange}
          controlConfig={controlConfig}
          onControlChange={onControlChange}
          availableSampleIds={availableSampleIds}
        />
      )}

      {/* Drug library (shown after input upload) */}
      {inputUpload && (
        <DrugLibraryPanel drugs={drugs} isLoading={isDrugLoading} onDrugClick={onDrugClick} />
      )}

      {/* Query generation (shown after input upload) */}
      {inputUpload && (
        <QueryGenerationPanel
          inputDegree={inputUpload.degree}
          onQueryGenerated={handleQueryGenerated}
        />
      )}
    </div>
  );
}
