import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { UploadResponse } from './types/silo1';
import DataInputPanel from './components/DataInputPanel';
import InferenceControls from './components/InferenceControls';
import DoseResponseSection from './components/DoseResponseSection';
import AnalyticsSection from './components/AnalyticsSection';
import CombinationSection from './components/CombinationSection';
import GuidedTour from './components/GuidedTour';
import { useInference } from './hooks/useInference';
import { useControlBaseline } from './hooks/useControlBaseline';
import { useDeltaComputation } from './hooks/useDeltaComputation';
import { VividContext } from './contexts/VividContext';
import type {
  SamplePredictionResult,
  InputDataPoint,
} from './types/dose-response';
import type { PredictionRow } from './types/api';
import './App.css';

const queryClient = new QueryClient();

function Dashboard() {
  // Vivid mode for high-contrast display
  const [vivid, setVivid] = useState(false);

  // Track whether data is ready for inference
  const [hasInput, setHasInput] = useState(false);
  const [hasQuery, setHasQuery] = useState(false);
  const [inputData, setInputData] = useState<InputDataPoint[]>([]);
  const [subsampledIndices, setSubsampledIndices] = useState<Set<number> | null>(null);

  // External zoom trigger (from heatmap/barplot clicks)
  const [externalZoom, setExternalZoom] = useState<{ drugKey: string; sampleId: string } | null>(null);

  // Upload response (for sample IDs, drug names)
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);

  // Analytics: group tracking
  const [groupColumn, setGroupColumn] = useState<string | null>(null);
  const [sampleGroups, setSampleGroups] = useState<Record<string, string>>({});

  // Single inference hook
  const inference = useInference();

  // Background MAE inference hook

  // Track whether MAE has been auto-started for the current inference result
  // Analytics hooks
  const { controlConfig, setControlSamples } = useControlBaseline();
  const { deltas, rankedHits, maeSummaries, flatPredictions } = useDeltaComputation(
    inference.partialResults,
    controlConfig,
  );

  // Extract metrics from final result
  const metrics = useMemo(() => {
    return (inference.result?.metrics as Record<string, { auc: number; ic50: number | null }>) || {};
  }, [inference.result]);

  // Stable sample map from inference (only affected samples get new refs)
  const sampleMap = inference.sampleMap;

  // Handlers
  const handleRun = useCallback(() => {
    inference.run('predict_full', {});
  }, [inference]);

  const handleStop = useCallback(() => {
    inference.cancel();
  }, [inference]);

  // Extract unique drug names from input data
  const drugNames = useMemo(() => {
    const names = new Set<string>();
    inputData.forEach(row => {
      if (row.drug1) names.add(row.drug1);
      if (row.drug2) names.add(row.drug2);
    });
    return Array.from(names).sort();
  }, [inputData]);

  const handleDrugClick = useCallback((drugKey: string, sampleId: string) => {
    setExternalZoom({ drugKey, sampleId });
  }, []);

  // Callback from DataInputPanel to notify when data changes
  const handleDataChange = useCallback(
    (input: boolean, query: boolean, inputRows?: InputDataPoint[]) => {
      setHasInput(input);
      setHasQuery(query);
      if (inputRows) {
        setInputData(inputRows);
      }
    },
    [],
  );

  // Callback when input is uploaded
  const handleInputReady = useCallback((response: UploadResponse) => {
    setUploadResponse(response);
    const mapping = response.column_mapping;
    const groupCol = mapping.group || null;
    setGroupColumn(groupCol);
  }, []);

  // Fetch group mapping from backend after input is uploaded
  useEffect(() => {
    if (!hasInput) return;

    async function fetchGroups() {
      try {
        const resp = await fetch('/api/session/groups');
        if (resp.ok) {
          const data = await resp.json();
          setSampleGroups(data.groups || {});
          if (data.group_column) {
            setGroupColumn(data.group_column);
          }
        }
      } catch {
        // Groups endpoint not available
      }
    }
    fetchGroups();
  }, [hasInput]);

  const canRun = hasInput && hasQuery;
  const hasResults = deltas.length > 0;

  return (
    <VividContext.Provider value={vivid}>
    <div className={`app${vivid ? ' vivid' : ''}`}>
      <header className="app-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>ScreenShot Dashboard</h1>
            <p>Foundation Model for Drug Screening</p>
          </div>
          <button
            onClick={async () => {
              await fetch('/api/session/reset', { method: 'POST' });
              window.location.reload();
            }}
            style={{
              background: 'none',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-sm)',
              padding: '4px 12px',
              fontSize: 'var(--text-sm)',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
            }}
          >
            Reset
          </button>
        </div>
      </header>
      <main className="app-main">
        <DataInputPanel
          onDataChange={handleDataChange}
          onInputReady={handleInputReady}
          onSubsampleIndicesChange={setSubsampledIndices}
          controlConfig={controlConfig}
          onControlChange={setControlSamples}
        />

        <InferenceControls
          isRunning={inference.isRunning}
          progress={inference.progress}
          progressCurrent={inference.progressCurrent}
          progressTotal={inference.progressTotal}
          onRun={handleRun}
          onStop={handleStop}
          canRun={canRun}
        />

        <div>
          <DoseResponseSection
            sampleMap={sampleMap}
            inputData={inputData}
            subsampledIndices={subsampledIndices}
            isLoading={inference.isRunning}
            progressCurrent={inference.progressCurrent}
            progressTotal={inference.progressTotal}
            controlConfig={controlConfig}
            metrics={metrics}
            externalZoom={externalZoom}
            onExternalZoomClear={() => setExternalZoom(null)}
          />
        </div>

        {/* Analytics Section */}
        {hasResults && (
          <AnalyticsSection
            deltas={deltas}
            rankedHits={rankedHits}
            maeSummaries={maeSummaries}
            flatPredictions={flatPredictions}
            controlConfig={controlConfig}
            groupColumn={groupColumn}
            sampleGroups={sampleGroups}
            onDrugClick={handleDrugClick}
          />
        )}

        {hasInput && drugNames.length > 1 && uploadResponse?.sample_ids?.length > 0 && (
          <CombinationSection
            sampleIds={uploadResponse.sample_ids}
            drugNames={drugNames}
          />
        )}

      </main>


    </div>
    </VividContext.Provider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}

export default App;
