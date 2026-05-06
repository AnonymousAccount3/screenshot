import { useCallback, useRef, useState } from 'react';
import type { UploadResponse } from '../types/silo1';
import { api } from '../api/client';

interface UploadZoneProps {
  type: 'input' | 'query';
  onUploadSuccess: (response: UploadResponse, fileName: string, csvText?: string) => void;
  onUploadError: (error: string) => void;
  uploadResponse: UploadResponse | null;
  fileName: string | null;
  error: string | null;
}

export default function UploadZone({
  type,
  onUploadSuccess,
  onUploadError,
  uploadResponse,
  fileName,
  error,
}: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setIsUploading(true);
      try {
        const name = file.name.toLowerCase();
        let response: UploadResponse;
        let csvText: string | undefined;
        if (name.endsWith('.feather') || name.endsWith('.xlsx') || name.endsWith('.xls')) {
          const buf = await file.arrayBuffer();
          const contentType = name.endsWith('.feather') ? 'application/octet-stream'
            : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
          response = await api.uploadInputBinary(buf, contentType);
        } else {
          csvText = await file.text();
          response = await api.uploadInput(csvText);
        }
        onUploadSuccess(response, file.name, csvText);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Upload failed';
        onUploadError(message);
      } finally {
        setIsUploading(false);
      }
    },
    [type, onUploadSuccess, onUploadError],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const testIdPrefix = type;
  const label = type === 'input' ? 'Input Data' : 'Query Data';

  return (
    <div
      data-testid={`${testIdPrefix}-upload-zone`}
      className={`upload-zone ${isDragging ? 'dragging' : ''} ${uploadResponse ? 'has-file' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      style={{
        border: `2px dashed ${isDragging ? 'var(--color-brand-400)' : 'var(--border-strong)'}`,
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--space-6) var(--space-5)',
        textAlign: 'center',
        backgroundColor: isDragging ? 'var(--color-brand-50)' : 'var(--surface-muted)',
        cursor: 'pointer',
        minHeight: '120px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 'var(--space-2)',
        transition: 'background-color 150ms ease, border-color 150ms ease',
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.feather,.xlsx,.xls"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {isUploading ? (
        <p style={{ color: 'var(--text-secondary)' }}>Uploading...</p>
      ) : uploadResponse ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', flexWrap: 'wrap', width: '100%', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
            <span style={{ fontSize: 'var(--text-lg)', color: 'var(--color-brand-500)' }}>&#10003;</span>
            <div style={{ textAlign: 'left' }}>
              <p data-testid={`${testIdPrefix}-file-info`} style={{ margin: 0, fontWeight: 'var(--font-weight-semibold)', fontSize: 'var(--text-base)' }}>
                {fileName === '(restored)' || fileName === 'cohort' ? 'Data loaded' : fileName}
              </p>
              <p data-testid={`${testIdPrefix}-row-count`} style={{ margin: '2px 0 0', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                {uploadResponse.row_count} rows &middot; {uploadResponse.columns.length} columns &middot; degree {uploadResponse.degree}
              </p>
            </div>
          </div>
          {uploadResponse.warnings.length > 0 && (
            <div style={{
              color: 'var(--color-warning-600)',
              fontSize: 'var(--text-xs)',
              backgroundColor: 'var(--color-warning-100)',
              padding: '4px 10px',
              borderRadius: 'var(--radius-sm)',
              maxWidth: '400px',
              textAlign: 'left',
            }}>
              {uploadResponse.warnings.map((w, i) => (
                <p key={i} style={{ margin: i === 0 ? 0 : '2px 0 0' }}>{w}</p>
              ))}
            </div>
          )}
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{ fontSize: 'var(--text-sm)' }}
          >
            Replace
          </button>
        </div>
      ) : (
        <>
          <p style={{ fontWeight: 'var(--font-weight-semibold)', fontSize: 'var(--text-md)' }}>{label}</p>
          <p style={{ color: 'var(--text-tertiary)', fontSize: 'var(--text-base)' }}>
            Drag & drop a file here, or click to browse (CSV, Feather, Excel)
          </p>
          <button onClick={() => fileInputRef.current?.click()}>
            Choose File
          </button>
        </>
      )}

      {error && (
        <div
          data-testid="upload-error"
          style={{
            color: 'var(--color-error-600)',
            backgroundColor: 'var(--color-error-100)',
            padding: 'var(--space-2) var(--space-3)',
            borderRadius: 'var(--radius-sm)',
            marginTop: 'var(--space-2)',
            fontSize: 'var(--text-base)',
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
