import { useState, useMemo, useRef, useEffect } from 'react';

const API_BASE = '/api';

interface CombinationSectionProps {
  sampleIds: string[];
  drugNames: string[];
}

export default function CombinationSection({ sampleIds, drugNames }: CombinationSectionProps) {
  const [sample, setSample] = useState('');
  const [drug1, setDrug1] = useState('');
  const [drug2, setDrug2] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    doses: number[];
    viability: number[];
    drug1: string;
    drug2: string;
    sample_id: string;
  } | null>(null);
  const plotRef = useRef<HTMLDivElement>(null);

  const canPredict = sample && drug1 && drug2 && drug1 !== drug2;

  const handlePredict = async () => {
    if (!canPredict) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/combination/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_id: sample, drug1, drug2, n_points: 10 }),
      });
      if (!resp.ok) {
        const body = await resp.json();
        throw new Error(body.detail || `Error ${resp.status}`);
      }
      setResult(await resp.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  // Render plot with dynamic plotly import
  useEffect(() => {
    if (!result || !plotRef.current) return;

    const n = result.doses.length;
    const z: number[][] = [];
    for (let i = 0; i < n; i++) {
      z.push(result.viability.slice(i * n, (i + 1) * n));
    }

    import('plotly.js-dist-min').then((Plotly) => {
      if (!plotRef.current) return;
      Plotly.newPlot(plotRef.current, [{
        type: 'surface',
        x: result.doses,
        y: result.doses,
        z,
        colorscale: [
          [0, '#1565c0'],
          [0.5, '#e8e8e8'],
          [1, '#c62828'],
        ],
        colorbar: { title: 'Viability', thickness: 15 },
        hovertemplate:
          `${result.drug1}: %{x:.1f}<br>` +
          `${result.drug2}: %{y:.1f}<br>` +
          'Viability: %{z:.3f}<extra></extra>',
      }], {
        width: 600,
        height: 500,
        margin: { l: 0, r: 0, t: 30, b: 0 },
        title: `${result.drug1} + ${result.drug2} — ${result.sample_id}`,
        scene: {
          xaxis: { title: `${result.drug1} (log\u2081\u2080 \u03BCM)` },
          yaxis: { title: `${result.drug2} (log\u2081\u2080 \u03BCM)` },
          zaxis: { title: 'Viability', range: [0, 1.2] },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
        },
      }, {
        displayModeBar: true,
        responsive: true,
      });
    });
  }, [result]);

  if (sampleIds.length === 0 || drugNames.length < 2) return null;

  return (
    <div className="card" style={{ marginTop: 'var(--space-4)' }}>
      <h3 style={{ marginTop: 0 }}>Combination Response Surface</h3>

      <div style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: 'var(--space-3)' }}>
        <div>
          <label style={{ display: 'block', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: '4px' }}>Sample</label>
          <select value={sample} onChange={e => setSample(e.target.value)} style={{ minWidth: '140px' }}>
            <option value="">Select...</option>
            {sampleIds.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: '4px' }}>Drug 1</label>
          <select value={drug1} onChange={e => setDrug1(e.target.value)} style={{ minWidth: '160px' }}>
            <option value="">Select...</option>
            {drugNames.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: '4px' }}>Drug 2</label>
          <select value={drug2} onChange={e => setDrug2(e.target.value)} style={{ minWidth: '160px' }}>
            <option value="">Select...</option>
            {drugNames.filter(d => d !== drug1).map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <button onClick={handlePredict} disabled={!canPredict || loading} style={{ padding: '6px 16px' }}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {error && <p style={{ color: 'var(--color-error-600)' }}>{error}</p>}

      <div ref={plotRef} />
    </div>
  );
}
