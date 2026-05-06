
import { useCallback, useEffect, useRef, useState } from 'react';
import type { WSMessage, TaskType } from '../types/api';

interface UseWebSocketOptions {
    onProgress?: (msg: WSMessage) => void;
    onComplete?: (msg: WSMessage) => void;
    onError?: (msg: WSMessage) => void;
    onCancelled?: (msg: WSMessage) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Build WS URL relative to current host
  const getWsUrl = useCallback(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws`;
  }, []);

  const connect = useCallback(() => {
    const ws = new WebSocket(getWsUrl());

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 2s
      setTimeout(() => connect(), 2000);
    };

    ws.onmessage = (event) => {
      const msg: WSMessage = JSON.parse(event.data);
      switch (msg.type) {
        case 'progress':
          optionsRef.current.onProgress?.(msg);
          break;
        case 'completed':
        case 'complete':
          optionsRef.current.onComplete?.(msg);
          setCurrentTaskId(null);
          break;
        case 'error':
        case 'failed':
          optionsRef.current.onError?.(msg);
          setCurrentTaskId(null);
          break;
        case 'cancelled':
          optionsRef.current.onCancelled?.(msg);
          setCurrentTaskId(null);
          break;
        case 'submitted':
          setCurrentTaskId(msg.task_id);
          break;
      }
    };

    wsRef.current = ws;
  }, [getWsUrl]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

    const submitTask = useCallback(
    (taskType: TaskType, params: Record<string, unknown> = {}) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      // Cancel current task first
      if (currentTaskId) {
        ws.send(JSON.stringify({ action: 'cancel', task_id: currentTaskId }));
      }

      ws.send(JSON.stringify({ action: 'submit', task_type: taskType, ...params }));
    },
    [currentTaskId],
  );

    const cancelTask = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !currentTaskId) return;
    ws.send(JSON.stringify({ action: 'cancel', task_id: currentTaskId }));
  }, [currentTaskId]);

  return { connected, currentTaskId, submitTask, cancelTask };
}
