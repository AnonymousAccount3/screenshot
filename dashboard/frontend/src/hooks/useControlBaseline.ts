
import { useState, useEffect, useCallback } from 'react';
import type { ControlConfig } from '../types/api';
import { api } from '../api/client';

interface UseControlBaselineReturn {
  controlConfig: ControlConfig;
  isLoading: boolean;
  error: string | null;
  setControlSamples: (sampleIds: string[]) => Promise<void>;
}

const DEFAULT_CONFIG: ControlConfig = {
  method: 'median',
  control_sample_ids: [],
};

export function useControlBaseline(): UseControlBaselineReturn {
  const [controlConfig, setControlConfig] = useState<ControlConfig>(DEFAULT_CONFIG);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchControls() {
      setIsLoading(true);
      try {
        const config = await api.getControls();
        if (!cancelled) {
          setControlConfig(config);
          setError(null);
        }
      } catch {
        if (!cancelled) {
          // Use default config if fetch fails (e.g. server not running in test mode)
          setControlConfig(DEFAULT_CONFIG);
          setError(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }
    fetchControls();
    return () => { cancelled = true; };
  }, []);

  const setControlSamples = useCallback(async (sampleIds: string[]) => {
    try {
      const config = await api.setControls(sampleIds);
      setControlConfig(config);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set controls');
    }
  }, []);

  return { controlConfig, isLoading, error, setControlSamples };
}
