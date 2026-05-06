import { useState, useCallback } from 'react';
import type { UploadResponse, ColumnMappingOverride } from '../types/silo1';
import type { ControlConfig } from '../types/api';

interface ColumnMappingPanelProps {
  uploadResponse: UploadResponse;
  onMappingChange: (newMapping: ColumnMappingOverride) => void;
  controlConfig: ControlConfig;
  onControlChange: (sampleIds: string[]) => void;
  availableSampleIds: string[];
}

const ROLE_LABELS: Record<string, string> = {
  sample_id: 'Sample ID',
  drug1: 'Drug 1',
  dose1: 'Dose 1',
  drug2: 'Drug 2',
  dose2: 'Dose 2',
  drug3: 'Drug 3',
  dose3: 'Dose 3',
  viability: 'Viability',
  group: 'Group',
};

// Compact inline mapping field
function MappingField({
  label,
  role,
  value,
  columns,
  onChange,
}: {
  label: string;
  role: string;
  value: string;
  columns: string[];
  onChange: (role: string, value: string) => void;
}) {
  return (
    <div
      data-testid={`mapping-${role}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
      }}
    >
      <label style={{ fontSize: 'var(--text-sm)', fontWeight: 'var(--font-weight-medium)', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {label}:
      </label>
      <select
        value={value}
        onChange={(e) => onChange(role, e.target.value)}
        style={{ fontSize: 'var(--text-sm)', padding: '4px 8px' }}
      >
        <option value="">—</option>
        {columns.map((col) => (
          <option key={col} value={col}>
            {col}
          </option>
        ))}
      </select>
    </div>
  );
}

export default function ColumnMappingPanel({
  uploadResponse,
  onMappingChange,
  controlConfig,
  onControlChange,
  availableSampleIds,
}: ColumnMappingPanelProps) {
  const { columns, column_mapping } = uploadResponse;
  const [numDrugs, setNumDrugs] = useState(() => {
    let n = 1;
    if (column_mapping.drug2) n = 2;
    if (column_mapping.drug3) n = 3;
    return n;
  });
  const [showControlPicker, setShowControlPicker] = useState(false);

  const handleChange = useCallback(
    (role: string, value: string) => {
      const newMapping: ColumnMappingOverride = { ...column_mapping };
      if (value === '') {
        (newMapping as Record<string, string | undefined>)[role] = undefined;
      } else {
        (newMapping as Record<string, string>)[role] = value;
      }
      onMappingChange(newMapping);
    },
    [column_mapping, onMappingChange],
  );

  return (
    <div
      data-testid="column-mapping-panel"
      className="card"
      style={{
        marginTop: 'var(--space-3)',
      }}
    >
      <h3 style={{ margin: '0 0 var(--space-3) 0', fontSize: 'var(--text-md)', fontWeight: 'var(--font-weight-semibold)' }}>Column Mapping</h3>

      {/* Basic fields row */}
      <div
        style={{
          display: 'flex',
          gap: 'var(--space-4)',
          marginBottom: 'var(--space-3)',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        <MappingField
          label={ROLE_LABELS.sample_id}
          role="sample_id"
          value={column_mapping.sample_id || ''}
          columns={columns}
          onChange={handleChange}
        />
        <MappingField
          label={ROLE_LABELS.viability}
          role="viability"
          value={column_mapping.viability || ''}
          columns={columns}
          onChange={handleChange}
        />
        <MappingField
          label={ROLE_LABELS.group}
          role="group"
          value={column_mapping.group || ''}
          columns={columns}
          onChange={handleChange}
        />
      </div>

      {/* Drug slots */}
      <div style={{ marginBottom: 'var(--space-3)' }}>
        <h4 style={{ margin: '0 0 var(--space-2) 0', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--font-weight-medium)' }}>
          Drugs
        </h4>
        {Array.from({ length: numDrugs }).map((_, i) => {
          const drugNum = i + 1;
          const drugRole = `drug${drugNum}`;
          const doseRole = `dose${drugNum}`;
          return (
            <div
              key={drugNum}
              style={{
                display: 'flex',
                gap: 'var(--space-4)',
                marginBottom: 'var(--space-2)',
                flexWrap: 'wrap',
                alignItems: 'center',
              }}
            >
              <MappingField
                label={ROLE_LABELS[drugRole]}
                role={drugRole}
                value={column_mapping[drugRole] || ''}
                columns={columns}
                onChange={handleChange}
              />
              <MappingField
                label={ROLE_LABELS[doseRole]}
                role={doseRole}
                value={column_mapping[doseRole] || ''}
                columns={columns}
                onChange={handleChange}
              />
            </div>
          );
        })}

        {numDrugs < 3 && (
          <button
            onClick={() => setNumDrugs(numDrugs + 1)}
            style={{
              padding: '4px var(--space-3)',
              fontSize: 'var(--text-sm)',
            }}
          >
            + Add Drug
          </button>
        )}
      </div>

      {/* Control Baseline — prominent */}
      <div
        style={{
          marginTop: 'var(--space-2)',
          padding: 'var(--space-3) var(--space-4)',
          backgroundColor: 'var(--color-brand-50)',
          border: '2px solid var(--color-brand-200)',
          borderRadius: 'var(--radius-md)',
          display: 'inline-block',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
          <h4 style={{ margin: 0, fontSize: 'var(--text-md)', color: 'var(--color-brand-800)', fontWeight: 'var(--font-weight-semibold)' }}>
            Control Baseline
          </h4>
          <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-brand-700)' }}>
            {controlConfig.method === 'user_selected' && controlConfig.control_sample_ids.length > 0
              ? `Selected: ${controlConfig.control_sample_ids.join(', ')}`
              : 'Median (all samples)'}
          </span>
          <div style={{ display: 'flex', gap: 'var(--space-2)', marginLeft: 'auto' }}>
            {controlConfig.method === 'user_selected' && controlConfig.control_sample_ids.length > 0 && (
              <button
                onClick={() => { onControlChange([]); setShowControlPicker(false); }}
                style={{
                  padding: '4px var(--space-3)',
                  fontSize: 'var(--text-sm)',
                  color: 'var(--color-error-600)',
                }}
              >
                Reset to Median
              </button>
            )}
            <button
              onClick={() => setShowControlPicker(!showControlPicker)}
              style={{
                padding: '4px var(--space-3)',
                fontSize: 'var(--text-sm)',
                backgroundColor: showControlPicker ? 'var(--color-brand-100)' : 'var(--surface-card)',
                borderColor: 'var(--color-brand-400)',
                color: 'var(--color-brand-700)',
                fontWeight: 'var(--font-weight-medium)',
              }}
            >
              {showControlPicker ? 'Hide' : 'Select control samples'}
            </button>
          </div>
        </div>

        {showControlPicker && availableSampleIds.length > 0 && (
          <div
            style={{
              marginTop: 'var(--space-2)',
              maxHeight: '150px',
              overflowY: 'auto',
              padding: 'var(--space-2)',
              backgroundColor: 'var(--surface-card)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              flexWrap: 'wrap',
              gap: 'var(--space-2)',
            }}
          >
            {availableSampleIds.map((sid) => {
              const isSelected = controlConfig.control_sample_ids.includes(sid);
              return (
                <label
                  key={sid}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: 'var(--text-sm)',
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: isSelected ? 'var(--color-brand-100)' : 'var(--surface-muted)',
                    border: isSelected ? '1px solid var(--color-brand-400)' : '1px solid var(--border-default)',
                    cursor: 'pointer',
                    transition: 'all 100ms ease',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => {
                      const next = isSelected
                        ? controlConfig.control_sample_ids.filter((id) => id !== sid)
                        : [...controlConfig.control_sample_ids, sid];
                      onControlChange(next);
                    }}
                  />
                  {sid}
                </label>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
