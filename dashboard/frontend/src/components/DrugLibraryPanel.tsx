import { useState } from 'react';
import type { DrugInfo } from '../types/silo1';

const COLLAPSED_COUNT = 9; // ~3 rows at typical widths

interface DrugLibraryPanelProps {
  drugs: DrugInfo[];
  isLoading: boolean;
  onDrugClick?: (drugName: string) => void;
}

export default function DrugLibraryPanel({
  drugs,
  isLoading,
  onDrugClick,
}: DrugLibraryPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const canCollapse = drugs.length > COLLAPSED_COUNT;
  const visibleDrugs = expanded || !canCollapse ? drugs : drugs.slice(0, COLLAPSED_COUNT);

  return (
    <div
      data-testid="drug-library-panel"
      className="card"
      style={{ marginTop: 'var(--space-3)' }}
    >
      <h3 style={{ margin: '0 0 var(--space-3) 0', fontSize: 'var(--text-lg)', fontWeight: 'var(--font-weight-semibold)' }}>
        Drug Library
        {drugs.length > 0 && (
          <span style={{ fontWeight: 'var(--font-weight-normal)', fontSize: 'var(--text-sm)', color: 'var(--text-tertiary)', marginLeft: 'var(--space-2)' }}>
            ({drugs.length} drugs)
          </span>
        )}
      </h3>

      {isLoading ? (
        <p style={{ color: 'var(--text-tertiary)' }}>Loading drug mappings...</p>
      ) : drugs.length === 0 ? (
        <p style={{ color: 'var(--text-tertiary)' }}>No drugs found in uploaded data.</p>
      ) : (
        <>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 'var(--space-2)',
            }}
          >
            {visibleDrugs.map((drug) => (
              <div
                key={drug.name}
                data-testid="drug-row"
                style={{
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: 'var(--space-3)',
                  backgroundColor: 'var(--surface-card)',
                  boxShadow: 'var(--shadow-xs)',
                  flex: '0 1 auto',
                  minWidth: '200px',
                  maxWidth: '280px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 'var(--space-2)',
                  transition: 'box-shadow 150ms ease, border-color 150ms ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                  <span
                    data-testid="drug-name-link"
                    role="button"
                    tabIndex={0}
                    onClick={() => onDrugClick?.(drug.name)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        onDrugClick?.(drug.name);
                      }
                    }}
                    style={{
                      color: 'var(--color-brand-600)',
                      cursor: 'pointer',
                      fontWeight: 'var(--font-weight-semibold)',
                      textDecoration: 'underline',
                      textDecorationStyle: 'dotted' as const,
                      flex: 1,
                      transition: 'color 150ms ease',
                    }}
                  >
                    {drug.name}
                  </span>
                  <span
                    data-testid="drug-status"
                    style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      borderRadius: 'var(--radius-pill)',
                      fontSize: 'var(--text-xs)',
                      fontWeight: 'bold',
                      backgroundColor: drug.known ? 'var(--color-success-100)' : 'var(--color-warning-100)',
                      color: drug.known ? 'var(--color-success-600)' : 'var(--color-warning-600)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {drug.known ? 'Known' : 'New'}
                  </span>
                </div>
                {drug.canonical_name && (
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
                    <span style={{ fontWeight: 500 }}>Canonical:</span> {drug.canonical_name}
                  </div>
                )}
              </div>
            ))}
          </div>
          {canCollapse && (
            <button
              onClick={() => setExpanded(!expanded)}
              style={{
                marginTop: 'var(--space-2)',
                padding: '4px 12px',
                border: '1px solid var(--border-default)',
                borderRadius: '4px',
                backgroundColor: 'var(--surface-muted)',
                cursor: 'pointer',
                fontSize: 'var(--text-sm)',
                color: 'var(--text-secondary)',
              }}
            >
              {expanded ? 'Show less' : `Show all ${drugs.length} drugs`}
            </button>
          )}
        </>
      )}
    </div>
  );
}
