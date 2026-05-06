import { useCallback, useRef, useState } from 'react';
import { useWebSocket } from './useWebSocket';
import type { TaskType, WSMessage, PredictionRow } from '../types/api';

interface InferenceState {
  isRunning: boolean;
  progress: number;
  progressCurrent: number;
  progressTotal: number;
    sampleMap: Map<string, PredictionRow[]>;
    partialResults: Record<string, unknown>[];
  result: Record<string, unknown> | null;
  error: string | null;
}

export function useInference() {
  const [state, setState] = useState<InferenceState>({
    isRunning: false,
    progress: 0,
    progressCurrent: 0,
    progressTotal: 0,
    sampleMap: new Map(),
    partialResults: [],
    result: null,
    error: null,
  });

  const partialsRef = useRef<Record<string, unknown>[]>([]);
  const sampleMapRef = useRef<Map<string, PredictionRow[]>>(new Map());
  const rafRef = useRef<number | null>(null);
  const pendingStateRef = useRef<Partial<InferenceState> | null>(null);

  const flushState = useCallback(() => {
    rafRef.current = null;
    if (pendingStateRef.current) {
      const pending = pendingStateRef.current;
      pendingStateRef.current = null;
      setState((prev) => ({ ...prev, ...pending }));
    }
  }, []);

  const { connected, currentTaskId, submitTask, cancelTask } = useWebSocket({
    onProgress: (msg: WSMessage) => {
      // Append to legacy partials
      const newPartials = [...partialsRef.current];
      if (msg.data) {
        newPartials.push(msg.data);

        // Update stable sample map
        const sampleId = msg.data.sample_id as string;
        const preds = msg.data.predictions as PredictionRow[] | undefined;
        if (sampleId && preds) {
          const prev = sampleMapRef.current;
          const newMap = new Map(prev);
          const existing = prev.get(sampleId);
          // Create new array ref only for this sample
          newMap.set(sampleId, existing ? [...existing, ...preds] : [...preds]);
          sampleMapRef.current = newMap;
        }
      }
      partialsRef.current = newPartials;

      pendingStateRef.current = {
        isRunning: true,
        progress: msg.total ? (msg.current ?? 0) / msg.total : 0,
        progressCurrent: msg.current ?? 0,
        progressTotal: msg.total ?? 0,
        sampleMap: sampleMapRef.current,
        partialResults: newPartials,
      };
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(flushState);
      }
    },
    onComplete: (msg: WSMessage) => {
      const completedData = (msg.data as Record<string, unknown>) ?? null;

      // If no partials were received, reconstruct from completed data
      if (partialsRef.current.length === 0 && completedData?.predictions) {
        const predictions = completedData.predictions as Record<string, unknown>[];
        const bySample = new Map<string, PredictionRow[]>();
        const reconstructed: Record<string, unknown>[] = [];

        for (const pred of predictions) {
          const sid = pred.sample_id as string;
          if (!bySample.has(sid)) bySample.set(sid, []);
          bySample.get(sid)!.push(pred as PredictionRow);
        }
        for (const [sid, preds] of bySample) {
          reconstructed.push({ sample_id: sid, predictions: preds });
        }
        partialsRef.current = reconstructed;
        sampleMapRef.current = bySample;
      }

      setState((prev) => ({
        ...prev,
        isRunning: false,
        progress: 1,
        sampleMap: sampleMapRef.current,
        partialResults: partialsRef.current,
        result: completedData,
      }));
    },
    onError: (msg: WSMessage) => {
      setState((prev) => ({
        ...prev,
        isRunning: false,
        error: msg.detail ?? 'Unknown error',
      }));
    },
    onCancelled: () => {
      setState((prev) => ({
        ...prev,
        isRunning: false,
      }));
    },
  });

  const run = useCallback(
    (taskType: TaskType, params: Record<string, unknown> = {}) => {
      partialsRef.current = [];
      sampleMapRef.current = new Map();
      setState({
        isRunning: true,
        progress: 0,
        progressCurrent: 0,
        progressTotal: 0,
        sampleMap: new Map(),
        partialResults: [],
        result: null,
        error: null,
      });
      submitTask(taskType, params);
    },
    [submitTask],
  );

  const cancel = useCallback(() => {
    cancelTask();
    partialsRef.current = [];
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    pendingStateRef.current = null;
  }, [cancelTask]);

  return {
    ...state,
    connected,
    currentTaskId,
    run,
    cancel,
  };
}
