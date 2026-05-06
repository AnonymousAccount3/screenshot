
import { useCallback, useRef, useState, useEffect } from 'react';
import type { PredictionRow, MAESummary } from '../types/api';
import { flattenPartialResults, computeMAESummaries } from '../utils/analytics';

interface MAEInferenceState {
    isRunning: boolean;
    progress: number;
    progressCurrent: number;
    progressTotal: number;
    fullDataMAESummaries: MAESummary[];
    fullDataFlatPredictions: PredictionRow[];
    ready: boolean;
    error: string | null;
}

const INITIAL_STATE: MAEInferenceState = {
  isRunning: false,
  progress: 0,
  progressCurrent: 0,
  progressTotal: 0,
  fullDataMAESummaries: [],
  fullDataFlatPredictions: [],
  ready: false,
  error: null,
};

export function useMAEInference() {
  const [state, setState] = useState<MAEInferenceState>(INITIAL_STATE);

  const wsRef = useRef<WebSocket | null>(null);
  const partialsRef = useRef<Record<string, unknown>[]>([]);
  const taskIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  // Cleanup WebSocket connection
  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    taskIdRef.current = null;
  }, []);

    const reset = useCallback(() => {
    cleanup();
    partialsRef.current = [];
    setState(INITIAL_STATE);
  }, [cleanup]);

    const start = useCallback(async () => {
    // Don't start if already running
    if (wsRef.current || taskIdRef.current) return;

    partialsRef.current = [];
    setState({
      ...INITIAL_STATE,
      isRunning: true,
    });

    try {
      // Submit the MAE task via REST
      const resp = await fetch('/api/inference/start-mae', { method: 'POST' });
      if (!resp.ok) {
        const detail = await resp.text();
        if (mountedRef.current) {
          setState((prev) => ({ ...prev, isRunning: false, error: detail }));
        }
        return;
      }
      const taskInfo = await resp.json();
      const taskId: string = taskInfo.task_id;
      taskIdRef.current = taskId;

      // Open task-specific WebSocket for streaming
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${window.location.host}/ws/${taskId}`);
      wsRef.current = ws;

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return;
        const msg = JSON.parse(event.data);

        switch (msg.type) {
          case 'progress': {
            if (msg.data) {
              partialsRef.current = [...partialsRef.current, msg.data];
            }
            setState((prev) => ({
              ...prev,
              progress: msg.total ? (msg.current ?? 0) / msg.total : 0,
              progressCurrent: msg.current ?? 0,
              progressTotal: msg.total ?? 0,
            }));
            break;
          }
          case 'completed':
          case 'complete': {
            // If no streaming partials arrived, reconstruct from final data
            let finalPartials = partialsRef.current;
            if (finalPartials.length === 0 && msg.data?.predictions) {
              const predictions = msg.data.predictions as Record<string, unknown>[];
              const bySample = new Map<string, Record<string, unknown>[]>();
              for (const pred of predictions) {
                const sid = pred.sample_id as string;
                if (!bySample.has(sid)) bySample.set(sid, []);
                bySample.get(sid)!.push(pred);
              }
              finalPartials = Array.from(bySample.entries()).map(
                ([sid, preds]) => ({ sample_id: sid, predictions: preds }),
              );
            }

            const flat = flattenPartialResults(finalPartials);
            const maeSummaries = computeMAESummaries(flat);

            setState({
              isRunning: false,
              progress: 1,
              progressCurrent: msg.total ?? flat.length,
              progressTotal: msg.total ?? flat.length,
              fullDataFlatPredictions: flat,
              fullDataMAESummaries: maeSummaries,
              ready: true,
              error: null,
            });

            ws.close();
            wsRef.current = null;
            taskIdRef.current = null;
            break;
          }
          case 'cancelled':
            setState((prev) => ({ ...prev, isRunning: false }));
            ws.close();
            wsRef.current = null;
            taskIdRef.current = null;
            break;
          case 'error':
          case 'failed':
            setState((prev) => ({
              ...prev,
              isRunning: false,
              error: msg.detail || 'MAE inference failed',
            }));
            ws.close();
            wsRef.current = null;
            taskIdRef.current = null;
            break;
        }
      };

      ws.onerror = () => {
        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            isRunning: false,
            error: 'WebSocket connection error during MAE inference',
          }));
        }
        wsRef.current = null;
        taskIdRef.current = null;
      };
    } catch (e) {
      if (mountedRef.current) {
        setState((prev) => ({ ...prev, isRunning: false, error: String(e) }));
      }
    }
  }, []);

    const cancel = useCallback(() => {
    if (wsRef.current && taskIdRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ action: 'cancel', task_id: taskIdRef.current }),
      );
    }
    cleanup();
    partialsRef.current = [];
    setState(INITIAL_STATE);
  }, [cleanup]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [cleanup]);

  return {
    ...state,
    start,
    cancel,
    reset,
  };
}
