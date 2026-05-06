import { useState, useCallback } from 'react';
import type { QueryGenerateResponse } from '../types/silo1';
import { api } from '../api/client';

interface QueryGenerationPanelProps {
  inputDegree: number;
  onQueryGenerated: (response: QueryGenerateResponse) => void;
}

export default function QueryGenerationPanel({
  inputDegree,
  onQueryGenerated,
}: QueryGenerationPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [nPoints, setNPoints] = useState(10);
  const [isGenerating, setIsGenerating] = useState(false);
  const [queryResponse, setQueryResponse] = useState<QueryGenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setError(null);
    try {
      const response = await api.generateQuery({
        degree: inputDegree,
        n_points: nPoints,
      });
      setQueryResponse(response);
      onQueryGenerated(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  }, [inputDegree, nPoints, onQueryGenerated]);

  return (
    <div
      className="card"
      style={{ marginTop: 'var(--space-3)' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
        <h3 style={{ margin: 0, fontSize: 'var(--text-lg)', fontWeight: 'var(--font-weight-semibold)' }}>Query Generation</h3>
        <button
          data-testid="query-generate-toggle"
          onClick={() => setIsOpen(!isOpen)}
          style={{
            padding: '4px 12px',
            borderRadius: '4px',
            border: '1px solid var(--border-default)',
            backgroundColor: isOpen ? 'var(--color-brand-50)' : 'var(--surface-card)',
            cursor: 'pointer',
          }}
        >
          {isOpen ? 'Hide Generator' : 'Generate Query'}
        </button>
      </div>

      {isOpen && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
            <label htmlFor="n-points">
              Number of points:
            </label>
            <input
              id="n-points"
              data-testid="n-points-input"
              type="number"
              min={1}
              max={1000}
              value={nPoints}
              onChange={(e) => setNPoints(parseInt(e.target.value, 10) || 10)}
              style={{ width: '80px', padding: '4px 8px' }}
            />
            <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--text-sm)' }}>
              dose per drug
            </span>
          </div>

          <button
            data-testid="generate-query-btn"
            onClick={handleGenerate}
            disabled={isGenerating}
            style={{
              padding: 'var(--space-2) var(--space-4)',
              borderRadius: 'var(--radius-md)',
              border: 'none',
              backgroundColor: isGenerating ? 'var(--color-brand-300)' : 'var(--color-brand-400)',
              color: 'var(--text-on-brand)',
              cursor: isGenerating ? 'not-allowed' : 'pointer',
              fontWeight: 'var(--font-weight-semibold)',
              fontSize: 'var(--text-base)',
              transition: 'all 150ms ease',
            }}
          >
            {isGenerating ? 'Generating...' : 'Generate'}
          </button>

          {error && (
            <p style={{ color: 'var(--color-error-600)', marginTop: '8px' }}>{error}</p>
          )}

          {queryResponse && (
            <div data-testid="query-preview" style={{ marginTop: 'var(--space-3)' }}>
              <p style={{ fontWeight: 'var(--font-weight-semibold)' }}>
                Generated grid: {queryResponse.row_count} rows
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
