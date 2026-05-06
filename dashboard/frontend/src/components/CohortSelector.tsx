import { useEffect, useState, useCallback } from 'react';
import type { CohortInfo, UploadResponse } from '../types/silo1';
import { api } from '../api/client';

interface CohortSelectorProps {
  onCohortLoaded: (response: UploadResponse) => void;
}

export default function CohortSelector({ onCohortLoaded }: CohortSelectorProps) {
  const [cohorts, setCohorts] = useState<CohortInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingCohort, setLoadingCohort] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listCohorts()
      .then((resp) => setCohorts(resp.cohorts))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load cohorts'))
      .finally(() => setIsLoading(false));
  }, []);

  const handleLoad = useCallback(
    async (name: string) => {
      setLoadingCohort(name);
      setError(null);
      try {
        const response = await api.loadCohort(name);
        onCohortLoaded(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load cohort');
      } finally {
        setLoadingCohort(null);
      }
    },
    [onCohortLoaded],
  );

  if (isLoading) {
    return <p style={{ color: 'var(--text-tertiary)' }}>Loading cohorts...</p>;
  }

  if (cohorts.length === 0) {
    return null;
  }

  return (
    <div
      data-testid="cohort-selector"
      className="card"
      style={{ marginTop: 'var(--space-3)' }}
    >
      <h3 style={{ margin: '0 0 var(--space-3) 0', fontSize: 'var(--text-lg)', fontWeight: 'var(--font-weight-semibold)' }}>
        Or load a pre-built cohort
      </h3>

      {error && (
        <p style={{ color: 'var(--color-error-600)', marginBottom: 'var(--space-2)' }}>{error}</p>
      )}

      <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
        {cohorts.map((cohort) => (
          <div
            key={cohort.name}
            data-testid="cohort-card"
            style={{
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              padding: 'var(--space-3) var(--space-4)',
              minWidth: '200px',
              cursor: 'pointer',
              backgroundColor:
                loadingCohort === cohort.name ? 'var(--surface-muted)' : 'var(--surface-card)',
              boxShadow: 'var(--shadow-xs)',
              transition: 'box-shadow 150ms ease, transform 150ms ease',
            }}
            onClick={() => handleLoad(cohort.name)}
          >
            <p style={{ fontWeight: 'var(--font-weight-semibold)', margin: '0 0 var(--space-1) 0', fontSize: 'var(--text-md)' }}>
              {cohort.name.replace(/_/g, ' ')}
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-sm)', margin: '0 0 var(--space-1) 0' }}>
              {cohort.description}
            </p>
            <p style={{ color: 'var(--text-tertiary)', fontSize: 'var(--text-xs)', margin: 0 }}>
              {cohort.n_samples} samples, {cohort.n_drugs} drugs
            </p>
            {loadingCohort === cohort.name && (
              <p style={{ color: 'var(--color-brand-600)', fontSize: 'var(--text-xs)', marginTop: 'var(--space-1)' }}>
                Loading...
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
