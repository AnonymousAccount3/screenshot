import { useState, useEffect } from 'react';

interface DeviceInfo {
  current: string;
  available: string[];
}

interface InferenceControlsProps {
  isRunning: boolean;
  progress: number;
  progressCurrent: number;
  progressTotal: number;
  onRun: () => void;
  onStop: () => void;
  canRun: boolean;
}

export default function InferenceControls({
  isRunning,
  progress,
  progressCurrent,
  progressTotal,
  onRun,
  onStop,
  canRun,
}: InferenceControlsProps) {
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    fetch('/api/device').then(r => r.json()).then(setDevice).catch(() => {});
  }, []);

  const handleDeviceChange = async (newDevice: string) => {
    setSwitching(true);
    try {
      const resp = await fetch('/api/device', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device: newDevice }),
      });
      const data = await resp.json();
      setDevice(prev => prev ? { ...prev, current: data.current } : null);
    } catch {}
    setSwitching(false);
  };

  return (
    <div style={{ margin: 'var(--space-5) 0', display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
      {!isRunning ? (
        <>
          <button
            data-testid="run-inference-btn"
            onClick={onRun}
            disabled={!canRun || switching}
            style={{
              padding: 'var(--space-3) var(--space-6)',
              borderRadius: 'var(--radius-md)',
              border: 'none',
              backgroundColor: canRun && !switching ? 'var(--color-brand-400)' : 'var(--color-neutral-300)',
              color: 'var(--text-on-brand)',
              cursor: canRun && !switching ? 'pointer' : 'not-allowed',
              fontWeight: 'var(--font-weight-semibold)',
              fontSize: 'var(--text-md)',
              transition: 'all 150ms ease',
            }}
          >
            Run Inference
          </button>
          {device && (
            <select
              value={device.current}
              onChange={e => handleDeviceChange(e.target.value)}
              disabled={isRunning || switching}
              style={{ fontSize: 'var(--text-sm)', padding: '4px 8px' }}
            >
              {device.available.map(d => (
                <option key={d} value={d}>{d.toUpperCase()}</option>
              ))}
            </select>
          )}
          {switching && <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-tertiary)' }}>Loading model...</span>}
        </>
      ) : (
        <>
          <button
            data-testid="stop-inference-btn"
            onClick={onStop}
            style={{
              padding: 'var(--space-3) var(--space-6)',
              borderRadius: 'var(--radius-md)',
              border: 'none',
              backgroundColor: 'var(--color-error-600)',
              color: 'var(--text-on-brand)',
              cursor: 'pointer',
              fontWeight: 'var(--font-weight-semibold)',
              fontSize: 'var(--text-md)',
              transition: 'background-color 150ms ease',
            }}
          >
            Stop
          </button>
          <div
            data-testid="inference-progress"
            style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}
          >
            <div style={{
              flex: 1, height: '6px',
              backgroundColor: 'var(--color-neutral-150)',
              borderRadius: 'var(--radius-pill)',
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${Math.max(progress * 100, 2)}%`,
                height: '100%',
                backgroundColor: 'var(--color-brand-500)',
                borderRadius: 'var(--radius-pill)',
                transition: 'width 300ms ease',
              }} />
            </div>
            <span
              data-testid="inference-progress-text"
              style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}
            >
              {progressCurrent} / {progressTotal}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
